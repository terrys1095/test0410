import os
import re
import io
import random
import argparse
import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn.functional as nn_f
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image

# --- 根據您的專案架構引入模組 ---
from resnet import get_resnet
from mobilenet import MobileNetV2
from densenet import DenseNet      # 👈 引入我們剛剛改好的客製化 DenseNet
from generator import Autoencoder
from utils import set_random_seed

import warnings

# 💡 忽略 sklearn 對於「樣本數與類別數比例」的過度關心警告
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def create_model(model_name, num_classes):
    model_name = model_name.lower()
    if 'resnet' in model_name:
        size = int(model_name.replace('resnet', '')) if model_name != 'resnet' else 10
        return get_resnet(size=size, num_classes=num_classes)
    elif model_name == 'mobilenet':
        return MobileNetV2(n_classes=num_classes)
    elif model_name == 'densenet':
        # 💡 改成使用本地的 DenseNet，並傳入正確的類別數
        return DenseNet(nClasses=num_classes)
    else:
        raise ValueError("不支援的模型！")

# ==========================================================
# ⚙️ 參數接收區 (支援外部腳本自動呼叫)
# ==========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--exp_root", type=str, default="experiment_results_collection_01", help="要評估的實驗總資料夾")
parser.add_argument("--csv_out", type=str, default="defense_sensitivity_analysis_01.csv", help="輸出的 CSV 檔名")
parser.add_argument("--dataset", type=str, default="cifar10", help="資料集: cifar10, cifar100, tinyimagenet") # 👈 新增參數
parser.add_argument("--model", type=str, default="resnet10", help="模型: resnet10, mobilenet, densenet") # 👈 新增這行
args = parser.parse_args()

EXPERIMENTS_ROOT = args.exp_root
CSV_OUTPUT_NAME = args.csv_out
MODEL_NAME = args.model

DATASET_NAME = args.dataset.lower()
if DATASET_NAME == 'cifar10':
    NUM_CLASSES = 10
elif DATASET_NAME == 'cifar100':
    NUM_CLASSES = 100
elif DATASET_NAME == 'tinyimagenet':
    NUM_CLASSES = 200
else:
    raise ValueError("❌ 不支援的資料集！請選擇 cifar10, cifar100, 或 tinyimagenet")

DEFENSE_LEVELS = [50]  # 要測試的 JPEG 品質參數
# DEFENSE_LEVELS = list(range(5, 96))  # 這會產生 30 到 80 的所有整數

ATTACK_TARGET_CLASS = 0                    # 攻擊的目標類別
CLIENT_NUM = 100                           # 總 Client 數量
CLIENT_BATCH = 32                          # 測試時的 Batch Size
SEED = 2024                                # 亂數種子
IS_SHUFFLE = 0                             # 訓練時是否有開啟 shuffle (0=沒有)

# 💡 [新增] 檢測測試的樣本數限制 (None 代表使用全部 Local Data)
DETECTION_N = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PGD 攻擊參數 ---
PGD_EPSILON = 4. / 255.
PGD_ALPHA = 4. / 255.
PGD_ITER = 1

# ==========================================================
# 🛡️ PGD 與防禦函數
# ==========================================================
def pgd_attack(model, images, labels):
    adv_images = images.clone().detach() + torch.zeros_like(images).uniform_(-PGD_EPSILON, PGD_EPSILON)
    adv_images = torch.clamp(adv_images, min=0, max=1)
    
    with torch.enable_grad():
        for _ in range(PGD_ITER):
            adv_images.requires_grad = True
            outputs = model(adv_images)
            loss = nn_f.cross_entropy(outputs, labels)
            loss.backward()
            
            adv_images = adv_images + PGD_ALPHA * torch.sign(adv_images.grad)
            eta = torch.clamp(adv_images - images, min=-PGD_EPSILON, max=PGD_EPSILON)
            adv_images = torch.clamp(images + eta, min=0, max=1)
            adv_images = adv_images.detach().clone()
            
    return adv_images

def apply_jpeg_defense(images, quality=50):
    restored_list = []
    for img in images:
        pil_img = torchvision.transforms.ToPILImage()(img.cpu())
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        jpeg_pil = Image.open(buffer)
        restored_list.append(torchvision.transforms.ToTensor()(jpeg_pil))
    return torch.stack(restored_list).to(images.device)

def sort_experiment_keys(folder_name):
    bad_match = re.search(r'Bad(\d+)', folder_name)
    bad_num = int(bad_match.group(1)) if bad_match else -1
    rate_match = re.search(r'Rate([\d.]+)', folder_name)
    rate_num = float(rate_match.group(1)) if rate_match else -1.0
    return (bad_num, rate_num)

# ==========================================================
# 🚀 主程式
# ==========================================================
def run_analysis():
    print(f"🚀 啟動批次評估 (目標資料夾: {EXPERIMENTS_ROOT} -> 輸出: {CSV_OUTPUT_NAME})")
    
    if not os.path.exists(EXPERIMENTS_ROOT):
        print(f"❌ 找不到資料夾 '{EXPERIMENTS_ROOT}'，跳過此任務。")
        return

    exp_folders = [f for f in os.listdir(EXPERIMENTS_ROOT) if os.path.isdir(os.path.join(EXPERIMENTS_ROOT, f))]
    exp_folders.sort(key=sort_experiment_keys, reverse=True)

    # 💡 動態準備測試資料集
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)), 
        torchvision.transforms.ToTensor()
    ])
    
    print(f"📦 正在準備測試資料集: {DATASET_NAME.upper()} (類別數: {NUM_CLASSES})")
    if DATASET_NAME == 'cifar10':
        test_dataset = torchvision.datasets.CIFAR10('./data', train=False, download=False, transform=test_transform)
    elif DATASET_NAME == 'cifar100':
        test_dataset = torchvision.datasets.CIFAR100('./data', train=False, download=False, transform=test_transform)
    elif DATASET_NAME == 'tinyimagenet':
        test_dataset = torchvision.datasets.ImageFolder('./data/tiny-imagenet-200/val', transform=test_transform)

    # results = [] 

    for exp_name in exp_folders:
        exp_dir = os.path.join(EXPERIMENTS_ROOT, exp_name)
        print(f"\n" + "="*50)
        print(f"🧪 正在評估實驗: {exp_name}")

        # 💡 [新增] 每個實驗開始時，建立一個專屬的列表來收集數據
        exp_results = []
        
        generator_path = os.path.join(exp_dir, "trained_trigger_generator.pth")
        if not os.path.exists(generator_path):
            print(f"⚠️ 找不到 Generator，跳過此實驗！")
            continue
            
        generator = Autoencoder().to(DEVICE)
        generator.load_state_dict(torch.load(generator_path, map_location=DEVICE))
        generator.eval()

        indices_path = os.path.join(exp_dir, "client_test_data_indices.pt")
        client_test_data_indices = torch.load(indices_path, map_location="cpu", weights_only=False)

        available_models = {}
        for fname in os.listdir(exp_dir):
            if fname.endswith(".pth") and ("clean_client" in fname or "bad_attacker" in fname):
                match = re.search(r'_(\d+)\.pth', fname)
                if match:
                    available_models[int(match.group(1))] = os.path.join(exp_dir, fname)

        if not available_models:
            continue

        # 💡 動態帶入類別數
        # eval_model = get_resnet(size=10, num_classes=NUM_CLASSES).to(DEVICE)
        eval_model = create_model(MODEL_NAME, NUM_CLASSES).to(DEVICE)

        # ==========================================================
        # 💡 [新增] 準備代理模型 (Surrogate Model)，代表攻擊者手上的模型
        # ==========================================================
        surrogate_model = create_model(MODEL_NAME, NUM_CLASSES).to(DEVICE)
        surrogate_model_path = None
    
        # 從掃描到的模型中，找一個惡意攻擊者的模型來當作 Surrogate Model
        for fname in os.listdir(exp_dir): # 💡 改成 exp_dir
            if fname.endswith(".pth") and "bad_attacker" in fname:
                surrogate_model_path = os.path.join(exp_dir, fname) # 💡 改成 exp_dir
                break # 找到一個就可以跳出了

        if surrogate_model_path:
            surrogate_model.load_state_dict(torch.load(surrogate_model_path, map_location=DEVICE))
            surrogate_model.eval()
            print(f"✅ 成功載入代理模型 (黑箱攻擊基準): {surrogate_model_path}")
        else:
            # 如果沒存到 bad_attacker，也可以考慮讀取 global_model.pth
            print(f"⚠️ 找不到 bad_attacker 模型來計算 PGD，請確認目錄！")
        
        set_random_seed(SEED)
        client_mapping = list(range(CLIENT_NUM))
        if IS_SHUFFLE == 1:
            random.shuffle(client_mapping)
        client_test_dataloaders = [
            torch.utils.data.DataLoader(test_dataset, batch_size=CLIENT_BATCH, 
                                        sampler=SubsetRandomSampler(client_test_data_indices[i]), drop_last=True) 
            for i in range(CLIENT_NUM)
        ]

        # 💡 [修改點] 準備收集該實驗所有 Client 的數據
        exp_base_acc_list = []
        exp_base_asr_list = []
        exp_base_precision_list = []
        exp_base_recall_list = []
        exp_base_f1_list = []
        exp_def_acc_dict = {q: [] for q in DEFENSE_LEVELS}
        exp_def_asr_dict = {q: [] for q in DEFENSE_LEVELS}
        exp_def_precision_dict = {q: [] for q in DEFENSE_LEVELS} # 👈 [新增]
        exp_def_recall_dict = {q: [] for q in DEFENSE_LEVELS}    # 👈 [新增]
        exp_def_f1_dict = {q: [] for q in DEFENSE_LEVELS}        # 👈 [新增]

        # 💡 [新增] 準備收集各 JPEG 品質下的「二元檢測 (Detection)」成績單
        exp_det_acc_dict = {q: [] for q in DEFENSE_LEVELS}
        exp_det_p_dict = {q: [] for q in DEFENSE_LEVELS}
        exp_det_r_dict = {q: [] for q in DEFENSE_LEVELS}
        exp_det_f1_dict = {q: [] for q in DEFENSE_LEVELS}

        loop = tqdm(sorted(available_models.keys()), desc="評估 Clients")
        for cid in loop:
            model_path = available_models[cid]
            eval_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            eval_model.eval()

            original_idx = client_mapping[cid]
            loader = client_test_dataloaders[original_idx]
            
            c_tot_clean, c_tot_poison = 0, 0
            c_corr_base, c_asr_base = 0, 0
            c_corr_def = {q: 0 for q in DEFENSE_LEVELS}
            c_asr_def = {q: 0 for q in DEFENSE_LEVELS}

            # 💡 [新增] 收集標籤與預測結果
            c_labels_all = []
            c_preds_base_all = []
            c_preds_def_all = {q: [] for q in DEFENSE_LEVELS}

            # 💡 [新增] 檢測任務專用的收集器與計數器
            c_det_true = [] 
            c_det_pred = {q: [] for q in DEFENSE_LEVELS}
            c_tested_n = 0

            for images, labels in loader:
                # 💡 [新增] 檢查是否達到 N 張圖片
                if DETECTION_N is not None and c_tested_n >= DETECTION_N:
                    break
                if DETECTION_N is not None and c_tested_n + len(labels) > DETECTION_N:
                    limit = DETECTION_N - c_tested_n
                    images, labels = images[:limit], labels[:limit]
                
                c_tested_n += len(labels)
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                c_tot_clean += len(labels)
                c_tot_poison += len(labels)
                
                # 💡 [新增] 記錄這批測試的真實標籤 (前半是乾淨圖=0，後半是惡意圖=1)
                c_det_true.extend([0] * len(labels) + [1] * len(labels))
                
                with torch.no_grad():
                    preds_base = eval_model(images).argmax(dim=1)
                    c_corr_base += (preds_base == labels).sum().item()

                    # 💡 [新增] 收集 Baseline
                    c_labels_all.extend(labels.cpu().numpy())
                    c_preds_base_all.extend(preds_base.cpu().numpy())

                # 💡 [關鍵修改] 丟入 surrogate_model 算梯度，模擬攻擊者只能用自己的模型算雜訊
                adv_imgs = pgd_attack(surrogate_model, images, labels)
                with torch.no_grad():
                    trigger = generator(images) / 255.0 * 4.0
                poisoned_images = torch.clamp(adv_imgs + trigger, 0, 1)

                with torch.no_grad():
                    preds_asr_base = eval_model(poisoned_images).argmax(dim=1)
                    c_asr_base += (preds_asr_base == ATTACK_TARGET_CLASS).sum().item()

                for qual in DEFENSE_LEVELS:
                    def_clean = apply_jpeg_defense(images, quality=qual)
                    with torch.no_grad():
                        preds_def_clean = eval_model(def_clean).argmax(dim=1)
                        c_corr_def[qual] += (preds_def_clean == labels).sum().item()
                        c_preds_def_all[qual].extend(preds_def_clean.cpu().numpy())

                    def_poison = apply_jpeg_defense(poisoned_images, quality=qual)
                    with torch.no_grad():
                        preds_def_asr = eval_model(def_poison).argmax(dim=1)
                        c_asr_def[qual] += (preds_def_asr == ATTACK_TARGET_CLASS).sum().item()

                    # 💡 [新增] 收集這一個 qual (防禦品質) 下的檢測預測結果
                    # 如果原圖與處理後的預測不同，標記為 1 (代表判定為惡意)
                    det_pred_clean = (preds_base != preds_def_clean).cpu().numpy().astype(int)
                    det_pred_poison = (preds_asr_base != preds_def_asr).cpu().numpy().astype(int)
                    c_det_pred[qual].extend(np.concatenate([det_pred_clean, det_pred_poison]))

            # 收集單一 Client 算出的百分比
            if c_tot_clean > 0:
                exp_base_acc_list.append((c_corr_base / c_tot_clean) * 100)
                exp_base_asr_list.append((c_asr_base / c_tot_poison) * 100)

                # 💡 [新增] 計算並收集 Base 指標
                p_base = precision_score(c_labels_all, c_preds_base_all, average='macro', zero_division=0) * 100
                r_base = recall_score(c_labels_all, c_preds_base_all, average='macro', zero_division=0) * 100
                f1_b = f1_score(c_labels_all, c_preds_base_all, average='macro', zero_division=0) * 100
                exp_base_precision_list.append(p_base)
                exp_base_recall_list.append(r_base)
                exp_base_f1_list.append(f1_b)

                for qual in DEFENSE_LEVELS:
                    exp_def_acc_dict[qual].append((c_corr_def[qual] / c_tot_clean) * 100)
                    exp_def_asr_dict[qual].append((c_asr_def[qual] / c_tot_poison) * 100)

                    # 💡 [新增] 計算並收集 Defense 指標
                    p_def = precision_score(c_labels_all, c_preds_def_all[qual], average='macro', zero_division=0) * 100
                    r_def = recall_score(c_labels_all, c_preds_def_all[qual], average='macro', zero_division=0) * 100
                    f1_d = f1_score(c_labels_all, c_preds_def_all[qual], average='macro', zero_division=0) * 100
                    exp_def_precision_dict[qual].append(p_def)
                    exp_def_recall_dict[qual].append(r_def)
                    exp_def_f1_dict[qual].append(f1_d)

                    # 💡 [新增] 結算該防禦強度的二元檢測 (Detection) 指標
                    det_acc = accuracy_score(c_det_true, c_det_pred[qual]) * 100
                    det_p = precision_score(c_det_true, c_det_pred[qual], pos_label=1, average='binary', zero_division=0) * 100
                    det_r = recall_score(c_det_true, c_det_pred[qual], pos_label=1, average='binary', zero_division=0) * 100
                    det_f1 = f1_score(c_det_true, c_det_pred[qual], pos_label=1, average='binary', zero_division=0) * 100
                    
                    exp_det_acc_dict[qual].append(det_acc)
                    exp_det_p_dict[qual].append(det_p)
                    exp_det_r_dict[qual].append(det_r)
                    exp_det_f1_dict[qual].append(det_f1)

        # 💡 [修改點] 寫入 CSV 的欄位擴充 (加入平均與標準差)
        if exp_base_acc_list:
            # --- 計算 Base 的平均值 (Avg) 與 標準差 (Std) ---
            avg_base_acc, std_base_acc = np.mean(exp_base_acc_list), np.std(exp_base_acc_list)
            avg_base_asr, std_base_asr = np.mean(exp_base_asr_list), np.std(exp_base_asr_list)
            avg_base_p, std_base_p = np.mean(exp_base_precision_list), np.std(exp_base_precision_list)
            avg_base_r, std_base_r = np.mean(exp_base_recall_list), np.std(exp_base_recall_list)
            avg_base_f1, std_base_f1 = np.mean(exp_base_f1_list), np.std(exp_base_f1_list)

            for qual in DEFENSE_LEVELS:
                # --- 計算 防禦後(Defended) 的平均值 (Avg) 與 標準差 (Std) ---
                avg_def_acc, std_def_acc = np.mean(exp_def_acc_dict[qual]), np.std(exp_def_acc_dict[qual])
                avg_def_asr, std_def_asr = np.mean(exp_def_asr_dict[qual]), np.std(exp_def_asr_dict[qual])
                avg_def_p, std_def_p = np.mean(exp_def_precision_dict[qual]), np.std(exp_def_precision_dict[qual])
                avg_def_r, std_def_r = np.mean(exp_def_recall_dict[qual]), np.std(exp_def_recall_dict[qual])
                avg_def_f1, std_def_f1 = np.mean(exp_def_f1_dict[qual]), np.std(exp_def_f1_dict[qual])

                # 💡 [新增] 計算 Detection 的平均與標準差
                avg_det_acc, std_det_acc = np.mean(exp_det_acc_dict[qual]), np.std(exp_det_acc_dict[qual])
                avg_det_p, std_det_p = np.mean(exp_det_p_dict[qual]), np.std(exp_det_p_dict[qual])
                avg_det_r, std_det_r = np.mean(exp_det_r_dict[qual]), np.std(exp_det_r_dict[qual])
                avg_det_f1, std_det_f1 = np.mean(exp_det_f1_dict[qual]), np.std(exp_det_f1_dict[qual])
                
                # 將結果寫入 List 中準備匯出成 CSV
                exp_results.append({
                    "Experiment": exp_name,
                    "JPEG_Quality": qual,
                    
                    "Avg_Base_ACC (%)": round(avg_base_acc, 2),
                    "Std_Base_ACC": round(std_base_acc, 2),            # 👈 [新增]
                    "Avg_Defended_ACC (%)": round(avg_def_acc, 2),
                    "Std_Defended_ACC": round(std_def_acc, 2),         # 👈 [新增]
                    "ACC_Drop (%)": round(avg_base_acc - avg_def_acc, 2),
                    
                    "Avg_Base_Precision (%)": round(avg_base_p, 2),
                    "Std_Base_Precision": round(std_base_p, 2),        # 👈 [新增]
                    "Avg_Defended_Precision (%)": round(avg_def_p, 2),
                    "Std_Defended_Precision": round(std_def_p, 2),     # 👈 [新增]
                    
                    "Avg_Base_Recall (%)": round(avg_base_r, 2),
                    "Std_Base_Recall": round(std_base_r, 2),           # 👈 [新增]
                    "Avg_Defended_Recall (%)": round(avg_def_r, 2),
                    "Std_Defended_Recall": round(std_def_r, 2),        # 👈 [新增]
                    
                    "Avg_Base_F1 (%)": round(avg_base_f1, 2),
                    "Std_Base_F1": round(std_base_f1, 2),              # 👈 [新增]
                    "Avg_Defended_F1 (%)": round(avg_def_f1, 2),
                    "Std_Defended_F1": round(std_def_f1, 2),           # 👈 [新增]
                    
                    "Avg_Base_ASR (%)": round(avg_base_asr, 2),
                    "Std_Base_ASR": round(std_base_asr, 2),            # 👈 [新增]
                    "Avg_Defended_ASR (%)": round(avg_def_asr, 2),
                    "Std_Defended_ASR": round(std_def_asr, 2),         # 👈 [新增]
                    "ASR_Drop (%)": round(avg_base_asr - avg_def_asr, 2),

                    # 💡 [新增] 寫入 Detection 指標
                    "Avg_Detection_ACC (%)": round(avg_det_acc, 2),
                    "Std_Detection_ACC": round(std_det_acc, 2),
                    "Avg_Detection_Precision (%)": round(avg_det_p, 2),
                    "Std_Detection_Precision": round(std_det_p, 2),
                    "Avg_Detection_Recall (%)": round(avg_det_r, 2),
                    "Std_Detection_Recall": round(std_det_r, 2),
                    "Avg_Detection_F1 (%)": round(avg_det_f1, 2),
                    "Std_Detection_F1": round(std_det_f1, 2)
                })

                
        # 💡 [新增] 在單一實驗 (exp_name) 測完後，立刻將結果存成專屬的 CSV
        if exp_results:
            # 1. 從您指定的 CSV_OUTPUT_NAME 中擷取出「資料夾路徑」，並自動建立
            out_dir = os.path.dirname(CSV_OUTPUT_NAME) if os.path.dirname(CSV_OUTPUT_NAME) else "."
            os.makedirs(out_dir, exist_ok=True)
            
            # 2. 組合出完美的檔名 (母資料夾_子資料夾.csv)
            root_name = os.path.basename(os.path.normpath(EXPERIMENTS_ROOT))
            dynamic_csv_name = f"{root_name}_{exp_name}.csv"
            
            # 3. 組合完整的儲存路徑
            save_path = os.path.join(out_dir, dynamic_csv_name)

            # 4. 寫入 CSV
            df = pd.DataFrame(exp_results)
            df = df.sort_values(by=['JPEG_Quality'])
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"\n🎉 實驗 {exp_name} 評估完成！資料已安全儲存至: {save_path}")
        else:
            print(f"\n⚠️ 實驗 {exp_name} 沒有收集到任何數據。")

    # if results:
    #     os.makedirs(os.path.dirname(CSV_OUTPUT_NAME) if os.path.dirname(CSV_OUTPUT_NAME) else ".", exist_ok=True)
    #     df = pd.DataFrame(results)
        
    #     df['Experiment'] = pd.Categorical(df['Experiment'], categories=exp_folders, ordered=True)
    #     # 排序：先實驗參數，再 JPEG 品質
    #     df = df.sort_values(by=['Experiment', 'JPEG_Quality'])
        
    #     df.to_csv(CSV_OUTPUT_NAME, index=False, encoding='utf-8-sig')
    #     print(f"\n🎉 評估完成！精華資料已儲存至: {CSV_OUTPUT_NAME}")
    # else:
    #     print("\n❌ 沒有收集到任何數據。")

if __name__ == "__main__":
    run_analysis()