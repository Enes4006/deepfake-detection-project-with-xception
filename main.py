import argparse
import os
from src.preprocess import extract_faces
from src.split_data import split_processed_data
from src.train import train_model

def main():
    parser = argparse.ArgumentParser(description="Bozok Deepfake Detection Platform - Xception Pipeline")
    parser.add_argument('--mode', type=str, default='train', choices=['preprocess', 'split', 'train', 'all'],
                        help="Çalıştırma modu: preprocess (yüz kesme), split (veri bölme), train (eğitim), all (hepsi sırayla)")
    
    args = parser.parse_args()
    
    if args.mode == 'preprocess':
        extract_faces()
        
    elif args.mode == 'split':
        split_processed_data()
        
    elif args.mode == 'train':
        train_model()
        
    elif args.mode == 'all':
        print("🏁 Tam otomatik süreç başlatılıyor...")
        extract_faces()
        split_processed_data()
        train_model()
        print("🎉 Tüm süreçler başarıyla tamamlandı! Model checkpoints/xception altında hazır.")

if __name__ == "__main__":
    main()