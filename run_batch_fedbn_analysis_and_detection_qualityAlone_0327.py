import os
import subprocess

def main():
    print("==================================================")
    print("🚀 開始全自動批次評估流程 (目標: 01 ~ 10)")
    print("==================================================")

    # 迴圈從 1 跑到 10
    for i in range(1, 11): # range(1, 6) range(1, 11) range(1, 2) range(1, 5)
        # 格式化數字，例如 1 變成 "01"，10 變成 "10"
        exp_idx = f"{i:02d}"
        
        # 組合資料夾名稱與輸出的 CSV 名稱
        # ==========aggregation=========  # range(1, 6)        
        exp_folder = f"TinyImageNet_0410/experiment_results_collection_fedbn_{exp_idx}"
        csv_file = f"defense_sensitivity_anyalsis_FedBN_csv_0410/defense_sensitivity_analysis_fedpac_{exp_idx}.csv"
        
        # ==========dataset=========記得調整資料集設定! # range(1, 2)
        # exp_folder = f"experiment_results_collection_datasetChange_0309/cifar100/experiment_results_collection_test1000/test"
        # csv_file = f"defense_sensitivity_anyalsis_FedAll_csv_0326/datasetChange/cifar100/defense_sensitivity_analysis_cifar100_{exp_idx}.csv"
        
        # exp_folder = f"datasetTinyImagenet1000_0309/experiment_results_collection_tinyImage_19"
        # csv_file = f"defense_sensitivity_anyalsis_FedAll_csv_0326/datasetChange/tinyImagenet/defense_sensitivity_analysis_tinyImagenet_{exp_idx}.csv"

        # ==========model=========記得調整資料集設定! # range(1, 2)
        # exp_folder = f"mobilenet1000_0310/experiment_results_collection_mobilenet_17__"
        # csv_file = f"defense_sensitivity_anyalsis_FedAll_csv_0326/modelChange/mobilenet/defense_sensitivity_analysis_mobilenet_{exp_idx}.csv"
        
        # exp_folder = f"densenet1000_0310/experiment_results_collection_densenetv2_09"
        # csv_file = f"defense_sensitivity_anyalsis_FedAll_csv_0326/modelChange/densenet/defense_sensitivity_analysis_densenet_{exp_idx}.csv"
        
        # ==========parameter=========記得調整資料集設定!
        # exp_folder = f"parameterChange0317/fedavg1000pC_0317/experiment_results_collection_fedavg_{exp_idx}" # range(1, 6)
        # csv_file = f"defense_sensitivity_anyalsis_FedAll_csv_0327/parameterChange/fedavg/defense_sensitivity_analysis_fedavg_{exp_idx}.csv"

        # exp_folder = f"experiment_results_collection_0322/fedavgTest_02" # range(1, 2)
        # csv_file = f"defense_sensitivity_anyalsis_FedAll_csv_0327/parameterChange/fedavg2/defense_sensitivity_analysis_fedavg_test1.csv"
        
        # exp_folder = f"parameterChange0317/ditto1000pC_0317/experiment_results_collection_ditto_{exp_idx}" # range(1, 3)
        # csv_file = f"defense_sensitivity_anyalsis_FedAll_csv_0325/parameterChange/ditto/defense_sensitivity_analysis_ditto_{exp_idx}.csv"

        # exp_folder = f"experiment_results_collection_fedbn_noShuffle0303/experiment_results_collection_{exp_idx}" # range(1, 11)
        # csv_file = f"defense_sensitivity_anyalsis_FedAll_csv_0325/parameterChange/fedbn/defense_sensitivity_analysis_fedbn_{exp_idx}.csv"
        




        print(f"\n▶️ 準備啟動任務 {exp_idx}/10 ...")
        print(f"📂 來源資料夾: {exp_folder}")
        print(f"📊 輸出檔案: {csv_file}")
        
        # 使用 subprocess 呼叫剛剛改好的 batch_evaluate.py，並傳遞參數
        cmd = [
            "python", "batch_evaluate_fed_and_detection_qualityAlone_0327.py", 
            "--exp_root", exp_folder, 
            "--csv_out", csv_file,
            "--dataset", "tinyimagenet",  # 根據您的實驗改為 cifar10, cifar100, 或 tinyimagenet
            "--model", "resnet152"    # 根據實驗替換為 resnet10, mobilenet, densenet
        ]
        
        try:
            # 執行指令，等它跑完才換下一個
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ 執行 {exp_folder} 時發生錯誤，將跳過並執行下一個。")
            continue

    print("\n" + "="*50)
    print("🎉 所有 10 個資料夾的評估任務皆已完成！")
    print("="*50)

if __name__ == "__main__":
    main()