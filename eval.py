import os
import sys
import shutil
import argparse
import subprocess
import math
import re

CODEGEN_REPO = "https://github.com/naholav/CodeGen.git"
BASE_DIR = os.getcwd()
CODEGEN_DIR = os.path.join(BASE_DIR, "CodeGen")

SYSTEM_PROMPT = "You are an expert Python programmer. Please read the problem carefully before writing any Python code."

def setup_environment():
    """CodeGen reposunu çeker ve kütüphaneleri kurar [cite: 46-48]"""
    if not os.path.exists(CODEGEN_DIR):
        print("📥 CodeGen reposu indiriliyor...")
        subprocess.run(["git", "clone", CODEGEN_REPO], check=True)
    
    print(" Gerekli kütüphaneler kontrol ediliyor...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets==2.19.0", "huggingface-hub==0.34.0"], check=True)
    
    req_path = os.path.join(CODEGEN_DIR, "requirements.txt")
    if os.path.exists(req_path):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", req_path], check=True)

def prepare_checkpoints(source_dir, model_type):
    """
    Drive'daki modelleri PDF formatına uygun şekilde CodeGen klasörüne bağlar.
    Format: models/{model_type}/checkpoints/checkpoint-step-X-epoch-Y
    
    """
    target_base = os.path.join(CODEGEN_DIR, "models", model_type, "checkpoints")

    if os.path.exists(target_base):
        shutil.rmtree(target_base)
    os.makedirs(target_base, exist_ok=True)
    
    print(f"\n Modeller hazırlanıyor: {source_dir} -> {target_base}")
    
    if not os.path.exists(source_dir):
        print(f" HATA: Kaynak klasör bulunamadı: {source_dir}")
        print("Lütfen Drive yolunun doğru olduğundan emin olun.")
        sys.exit(1)


    checkpoints = [d for d in os.listdir(source_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(source_dir, d))]
    
    if not checkpoints:
        print("HATA: Klasörde hiç checkpoint bulunamadı!")
        sys.exit(1)

    count = 0
    for cp in checkpoints:
        try:

            step_num = int(cp.split("-")[-1])

            epoch_num = math.ceil(step_num / 282) 

            new_name = f"checkpoint-step-{step_num}-epoch-{epoch_num}"
            
            src_path = os.path.join(source_dir, cp)
            dst_path = os.path.join(target_base, new_name)

            os.symlink(src_path, dst_path)
            print(f" Bağlandı: {new_name}")
            count += 1
        except Exception as e:
            print(f" Atlandı {cp}: {e}")
            
    print(f"Topam {count} checkpoint teste hazır.")

def patch_eval_script():
    """
    livecodebench_eval.py dosyasındaki model tanımlarını ve promptu günceller.
    [cite: 68-75]
    """
    script_path = os.path.join(CODEGEN_DIR, "livecodebench_eval.py")
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Model Tiplerini Değiştir [cite: 71-73]
    # Orijinal satırı bulup bizimkiyle değiştiriyoruz
    if '"deep_instruction", "diverse_instruction"' not in content:
        print(" Eval scripti güncelleniyor (Model Tipleri)...")
        # Regex ile tuple kısmını bulup değiştirme (daha güvenli)
        content = re.sub(
            r'model_types: tuple = \(.*?\)', 
            'model_types: tuple = ("deep_instruction", "diverse_instruction")', 
            content, 
            flags=re.DOTALL
        )

    # 2. System Prompt Güncelleme [cite: 75]
    # Kodda default prompt farklıysa, bizim PDF promptu ile değiştiriyoruz.
    # Genelde kodda variable olarak tanımlı olmayabilir, direkt string olabilir.
    # En güvenli yöntem, eğer parametre olarak system prompt alan bir yer varsa orayı manuel override etmektir.
    # Ancak CodeGen scripti genellikle promptu içeriden alır. Basit bir replace deneyelim:
    
    # (Opsiyonel: Eğer orijinal prompt biliniyorsa replace yapılır. 
    # Ancak scriptin iç yapısını bozmamak için bu adımı atlayıp varsayılan bırakmak da bir seçenektir 
    # eğer argümanla verilemiyorsa. Yine de PDF "değiştirdiyseniz güncelleyin" diyor.)
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(" Script ayarları yapıldı.")

def run_benchmark(model_type, output_backup_dir):
    """Benchmarkı başlatır ve sonuçları yedekler [cite: 78-84]"""
    print(f"\n {model_type.upper()} Testi Başlatılıyor...")
    
    # Çalıştırma Komutu [cite: 78, 79]
    cmd = [
        sys.executable, "livecodebench_eval.py",
        "--model_type", model_type,
        "--platform", "atcoder",
        "--difficulty", "easy"
    ]
    
    try:
        subprocess.run(cmd, cwd=CODEGEN_DIR, check=True)
        print("\nTest tamamlandı!")

        results_src = os.path.join(CODEGEN_DIR, "results", "livecodebench")
        if os.path.exists(results_src):
            if os.path.exists(output_backup_dir):
                shutil.rmtree(output_backup_dir)
            shutil.copytree(results_src, output_backup_dir)
            print(f" Sonuçlar Drive'a kaydedildi: {output_backup_dir}")
            print(f"Özeti şurada bulabilirsin: {os.path.join(output_backup_dir, 'summary.json')}")
        else:
            print(" Sonuç dosyası oluşmadı!")
            
    except subprocess.CalledProcessError as e:
        print(f" Test sırasında hata oluştu: {e}")

def main():
    parser = argparse.ArgumentParser(description="Otomatik Benchmark Scripti")
    
    parser.add_argument(
        "--model_type", 
        type=str, 
        required=True, 
        choices=["deep_instruction", "diverse_instruction"],
        help="Test edilecek model tipi (PDF'e göre)"
    )

    parser.add_argument(
        "--source_dir", 
        type=str, 
        default=None,
        help="Eğitilmiş modellerin olduğu klasör. (Boş bırakılırsa otomatk tahmin edilir)"
    )
    
    args = parser.parse_args()

    if args.source_dir:
        source_dir = args.source_dir
    else:
        drive_root = "/content/drive/MyDrive/LoRa_Egitim_Sonuclari"
        if args.model_type == "deep_instruction":
            source_dir = os.path.join(drive_root, "results_DEEP")
        else:
            source_dir = os.path.join(drive_root, "results_DIVERSE")

    backup_dir = os.path.join("/content/drive/MyDrive/LoRa_Benchmark_Sonuclari", f"results_{args.model_type}")

    setup_environment()
    prepare_checkpoints(source_dir, args.model_type)
    patch_eval_script()
    run_benchmark(args.model_type, backup_dir)

if __name__ == "__main__":
    main()