import subprocess
import time

def main():
    # 我們要跑第 2 次到第 10 次 (range 不包含結尾，所以寫 11)
    # 如果您連第 1 次都想重跑，可以改成 range(1, 11)
    start_run = 1
    end_run = 11 

    for i in range(start_run, end_run + 1):
        
        # 1. 定義資料夾名稱
        # dir_name = f"parameterChange0317/fedpac1000pC_0328/experiment_results_collection_fedpac_{i:02d}"
        dir_name = f"TinyImageNet_0410/experiment_results_collection_fedbn_{i:02d}"
        
        # 2. 定義 Seed 偏移量 (確保每次實驗隨機性不同)
        # 第 2 次實驗 seed = 2024 + 2 = 2026
        seed_offset = i 
        
        print(f"\n" + "="*60)
        print(f"🚀 開始執行第 {i} 輪完整實驗 (Global Run {i})")
        print(f"📂 存檔位置: {dir_name}")
        print(f"🎲 Random Seed Offset: {seed_offset}")
        print("="*60)

        # 呼叫我們剛剛改好的 run_experiments.py
        cmd = [
            "python", "run_experiments.py",
            "--base_dir", dir_name,
            "--seed_offset", str(seed_offset)
        ]
        
        subprocess.run(cmd, check=True)
        
        print(f"✅ 第 {i} 輪實驗全部完成！")
        time.sleep(5) # 休息一下

if __name__ == "__main__":
    main()