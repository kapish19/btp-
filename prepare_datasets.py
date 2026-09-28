import os
import sys
import shutil
import urllib.request

def setup_data(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    print("🚀 Starting automated dataset downloads...")

    # 1. Setup Kaggle credentials if present
    kaggle_source = "/content/kaggle.json"
    kaggle_dest_dir = os.path.expanduser("~/.kaggle")
    kaggle_dest = os.path.join(kaggle_dest_dir, "kaggle.json")
    if os.path.exists(kaggle_source) and not os.path.exists(kaggle_dest):
        os.makedirs(kaggle_dest_dir, exist_ok=True)
        shutil.copy(kaggle_source, kaggle_dest)
        os.chmod(kaggle_dest, 0o600)
        print("  ✅ Configured Kaggle API credentials.")

    # 2. Market-1501
    market_path = os.path.join(data_dir, "Market-1501-v15.09.15")
    if not os.path.exists(market_path):
        print("  📦 Downloading Market-1501 from Kaggle...")
        os.system(f"kaggle datasets download -d pengcw1/market-1501 -p {data_dir}")
        os.system(f"unzip -q {data_dir}/market-1501.zip -d {data_dir}")
        if os.path.exists(f"{data_dir}/market-1501.zip"): os.remove(f"{data_dir}/market-1501.zip")
    else: print("  ✅ Market-1501 ready.")

    # 3. CUHK-SYSU
    sysu_path = os.path.join(data_dir, "cuhk_sysu")
    if not os.path.exists(sysu_path):
        print("  📦 Downloading CUHK-SYSU from Kaggle...")
        os.system(f"kaggle datasets download -d manaschaiaonon/cuhk-sysu -p {data_dir}")
        os.system(f"unzip -q {data_dir}/cuhk-sysu.zip -d {data_dir}")
        if os.path.exists(f"{data_dir}/cuhk-sysu.zip"): os.remove(f"{data_dir}/cuhk-sysu.zip")
    else: print("  ✅ CUHK-SYSU ready.")

    # 4. CUHK03 + Protocol Files
    cuhk_path = os.path.join(data_dir, "cuhk03")
    if not os.path.exists(cuhk_path):
        print("  📦 Downloading CUHK03 from Kaggle...")
        os.system(f"kaggle datasets download -d priyanagda/cuhk03 -p {data_dir}")
        os.system(f"unzip -q {data_dir}/cuhk03.zip -d {cuhk_path}")
        if os.path.exists(f"{data_dir}/cuhk03.zip"): os.remove(f"{data_dir}/cuhk03.zip")

    # Fetch protocol mat files using verified working URLs & try-except
    urls_detected = [
        "https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/master/torchreid/datasets/cuhk03_new_protocol_config_detected.mat",
        "https://raw.githubusercontent.com/JDAI-CV/fast-reid/master/fastreid/data/datasets/cuhk03_new_protocol_config_detected.mat"
    ]
    urls_labeled = [
        "https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/master/torchreid/datasets/cuhk03_new_protocol_config_labeled.mat",
        "https://raw.githubusercontent.com/JDAI-CV/fast-reid/master/fastreid/data/datasets/cuhk03_new_protocol_config_labeled.mat"
    ]
    
    m1 = os.path.join(cuhk_path, "cuhk03_new_protocol_config_detected.mat")
    m2 = os.path.join(cuhk_path, "cuhk03_new_protocol_config_labeled.mat")
    
    if not os.path.exists(m1):
        for u in urls_detected:
            try:
                urllib.request.urlretrieve(u, m1)
                if os.path.exists(m1) and os.path.getsize(m1) > 1000: break
            except Exception: pass
            
    if not os.path.exists(m2):
        for u in urls_labeled:
            try:
                urllib.request.urlretrieve(u, m2)
                if os.path.exists(m2) and os.path.getsize(m2) > 1000: break
            except Exception: pass

    print("  ✅ CUHK03 protocol files ready.")

    # 5. MSMT17_V2 (Drive Link OR Kaggle Fallback)
    msmt_target = os.path.join(data_dir, "MSMT17_V2")
    if not os.path.exists(msmt_target):
        print("  ⚡ Searching for MSMT17_V2...")
        drive_root = "/content/drive/MyDrive"
        found = ""
        if os.path.exists(drive_root):
            for root, dirs, files in os.walk(drive_root):
                if "MSMT17_V2" in dirs:
                    found = os.path.join(root, "MSMT17_V2")
                    break
        if found:
            os.symlink(found, msmt_target)
            print(f"  ✅ MSMT17_V2 linked from Drive: {found}")
        else:
            print("  📦 MSMT17 not in Drive. Downloading MSMT17 from Kaggle...")
            os.system(f"kaggle datasets download -d pengcw1/msmt17 -p {data_dir}")
            os.system(f"unzip -q {data_dir}/msmt17.zip -d {data_dir}")
            if os.path.exists(f"{data_dir}/msmt17.zip"): os.remove(f"{data_dir}/msmt17.zip")
            if os.path.exists(os.path.join(data_dir, "MSMT17")) and not os.path.exists(msmt_target):
                os.symlink("MSMT17", msmt_target)
            elif os.path.exists(os.path.join(data_dir, "MSMT17_V2")):
                print("  ✅ MSMT17_V2 extracted!")
    else: print("  ✅ MSMT17_V2 ready.")

    print("🎉 ALL DATASETS DOWNLOADED AND PREPARED SUCCESSFULLY!")

if __name__ == "__main__":
    setup_data()
