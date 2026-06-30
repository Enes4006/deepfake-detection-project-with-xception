'''
import os
import shutil
import random

def split_processed_data(base_dir="data/processed", train_ratio=0.7, val_ratio=0.15):
    """
    Kırpılmış yüz klasörlerini video bazında Train, Val ve Test olarak böler.
    """
    categories = ['Real', 'Fake']
    phases = ['train', 'val', 'test']
    
    # 1. Yeni klasör yapısını oluşturalım
    for phase in phases:
        for cat in categories:
            os.makedirs(os.path.join(base_dir, phase, cat), exist_ok=True)
            
    for cat in categories:
        cat_dir = os.path.join(base_dir, cat)
        if not os.path.exists(cat_dir) or phase in os.listdir(base_dir):
            continue # Eğer zaten bölünmüşse veya klasör yoksa atla
            
        # Video klasörlerini listele
        video_folders = [f for f in os.listdir(cat_dir) if os.path.isdir(os.path.join(cat_dir, f))]
        random.seed(42) # Her çalıştırmada aynı ayrımı yapsın diye sabitleyebilirsin
        random.shuffle(video_folders)
        
        # Sınırları hesapla
        total_vids = len(video_folders)
        train_end = int(total_vids * train_ratio)
        val_end = train_end + int(total_vids * val_ratio)
        
        # Klasörleri ayır
        train_vids = video_folders[:train_end]
        val_vids = video_folders[train_end:val_end]
        test_vids = video_folders[val_end:]
        
        # Dosyaları yeni yerlerine taşı/kopyala
        def move_files(vid_list, phase_name):
            for vid in vid_list:
                src_vid_path = os.path.join(cat_dir, vid)
                dst_vid_path = os.path.join(base_dir, phase_name, cat, vid)
                shutil.move(src_vid_path, dst_vid_path)
                
        move_files(train_vids, 'train')
        move_files(val_vids, 'val')
        move_files(test_vids, 'test')
        
        # Eski boş ana klasörü temizle
        try: os.rmdir(cat_dir)
        except: pass

    print("🎉 Veri seti başarıyla bölündü: %70 Eğitim, %15 Doğrulama, %15 Test.")

if __name__ == "__main__":
    split_processed_data()


# Verileri ayırırken en kritik mühendislik kuralı şudur: Aynı videoya ait kareler hem train hem test setine dağılmamalıdır. Yoksa model videoyu ezberler ve test başarısı sahte bir şekilde yüksek çıkar.
'''
import os
import shutil
import random

def split_processed_data(base_dir="data/processed", train_ratio=0.7, val_ratio=0.15):
    categories = ['Real', 'Fake']
    phases = ['train', 'val', 'test']
    
    # Hedef klasörleri oluştur
    for phase in phases:
        for cat in categories:
            os.makedirs(os.path.join(base_dir, phase, cat), exist_ok=True)
            
    for cat in categories:
        cat_dir = os.path.join(base_dir, cat)
        
        # Eğer ana Real/Fake klasörü yoksa veya içi boşsa atla
        if not os.path.exists(cat_dir) or len(os.listdir(cat_dir)) == 0:
            continue
            
        # Klasörün içindeki tüm dosyaları/klasörleri topla
        items = [f for f in os.listdir(cat_dir)]
        random.seed(42)
        random.shuffle(items)
        
        total_items = len(items)
        train_end = int(total_items * train_ratio)
        val_end = train_end + int(total_items * val_ratio)
        
        train_items = items[:train_end]
        val_items = items[train_end:val_end]
        test_items = items[val_end:]
        
        # Windows izin engellerini aşmak için kopyalama/taşıma fonksiyonu
        def move_items(item_list, phase_name):
            for item in item_list:
                src_path = os.path.join(cat_dir, item)
                dst_path = os.path.join(base_dir, phase_name, cat, item)
                try:
                    shutil.move(src_path, dst_path)
                except Exception as e:
                    # Eğer taşıma hata verirse kopyalamayı dene
                    if os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                        shutil.rmtree(src_path)
                    else:
                        shutil.copy2(src_path, dst_path)
                        os.remove(src_path)
                        
        move_items(train_items, 'train')
        move_items(val_items, 'val')
        move_items(test_items, 'test')
        
        print(f"✅ {cat} sınıfı bölündü. Eğitim: {len(train_items)}, Val: {len(val_items)}, Test: {len(test_items)}")

if __name__ == "__main__":
    split_processed_data()