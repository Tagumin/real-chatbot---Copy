import subprocess
import sys
import os

def main():
    print("=" * 60)
    print("🚀 MULAI AUTO-EVALUASI UNTUK 3 DATASET")
    print("=" * 60)

    # Argumen dasar sesuai permintaan
    base_cmd = [
        sys.executable, "evaluate.py",
        "--k", "1", "3", "5",
        "--max-retrieve", "5",
        "--out-dir", ".",
        "--stage", "full"
    ]

    # Definisi 3 dataset
    jobs = [
        {
            "name": "1. Immigration (Law)",
            "domain": "law",
            "dataset": os.path.join("evaluate domain", "evaluator_immigration")
        },
        {
            "name": "2. Culture (Bali)",
            "domain": "culture",
            "dataset": os.path.join("evaluate domain", "evaluator_bali.json")
        },
        {
            "name": "3. Penal (Law)",
            "domain": "law",
            "dataset": os.path.join("evaluate domain", "evaluator_penal.json")
        }
    ]

    for job in jobs:
        print(f"\n▶ Menjalankan {job['name']}...")
        
        # Susun argumen lengkap
        cmd = base_cmd + [
            "--domain", job["domain"],
            "--dataset", job["dataset"]
        ]
        
        print("  Command: " + " ".join(cmd))
        
        # Jalankan proses
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"❌ Terjadi kesalahan saat mengevaluasi {job['name']}.")
        else:
            print(f"✅ Selesai mengevaluasi {job['name']}.")

    print("\n" + "=" * 60)
    print("🎉 SEMUA EVALUASI SELESAI")
    print("File CSV hasil evaluasi (.csv) telah disimpan di folder saat ini (.).")
    print("=" * 60)

if __name__ == "__main__":
    main()
