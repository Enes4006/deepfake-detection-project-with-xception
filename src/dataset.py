'''
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, phase="train", transform=None):
        """
        phase: 'train', 'val' veya 'test' olmalıdır.
        Bu sayede katmanlı mimaride hangi verinin yükleneceği merkezi olarak yönetilir.
        """
        self.phase_dir = os.path.join(root_dir, phase)
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Sınıf etiketleri (0: Real, 1: Fake)
        classes = {'Real': 0, 'Fake': 1}
        
        if not os.path.exists(self.phase_dir):
            raise FileNotFoundError(f"❌ Hata: '{self.phase_dir}' dizini bulunamadı! Lütfen split_data.py betiğini çalıştırın.")

        # Sadece ilgili faza (train/val/test) ait klasörleri tarar
        for class_name, label in classes.items():
            class_dir = os.path.join(self.phase_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            # Video klasörlerinin içindeki kırpılmış yüz resimlerini ekle
            for video_folder in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video_folder)
                if os.path.isdir(video_path):
                    for img_name in os.listdir(video_path):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            full_path = os.path.join(video_path, img_name)
                            # Bozuk veya 0 KB olan imajları filtrele
                            if os.path.getsize(full_path) > 0:
                                self.image_paths.append(full_path)
                                self.labels.append(label)

        print(f"📊 [{phase.upper()}] Seti Hazır: {len(self.image_paths)} adet yüz görüntüsü yüklendi.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # OpenCV ile hızlı I/O okuması
            image = cv2.imread(img_path)
            if image is None: 
                raise ValueError("Boş Görüntü")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            try:
                # Yedek hat: PIL okuması
                image = np.array(Image.open(img_path).convert("RGB"))
            except Exception:
                # Dosya tamamen bozuksa eğitimi çökertmemek için dummy siyah resim dönülür
                image = np.zeros((299, 299, 3), dtype=np.uint8)
            
        label = self.labels[idx]
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, torch.tensor(label, dtype=torch.long)


def get_transforms(img_size=299, phase="train"):
    """
    Xception mimarisine özel veri artırma ve normalizasyon hattı.
    Eğitim (train) için veri artırma uygulanırken, val ve test için sadece resize ve normalizasyon yapılır.
    """
    if phase == "train":
        return A.Compose([
            A.Resize(img_size, img_size), # Xception için zorunlu 299x299 boyutlandırma
            A.HorizontalFlip(p=0.5),      # %50 ihtimalle yatay çevirme
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
            ], p=0.3),                    # Işık ve renk manipülasyonlarına karşı dayanıklılık
            A.OneOf([
                A.GaussianBlur(p=0.5),
                A.GaussNoise(p=0.5),
                A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5), # Deepfake sıkıştırma izleri simülasyonu
            ], p=0.3),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # Xception / Inception standardı (-1 ile 1 arası)
            ToTensorV2()
        ])
    else:
        # Doğrulama ve Test setlerinde veriyi manipüle etmiyoruz, sadece modele hazır hale getiriyoruz
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])
'''


'''
💡 Bu Kodda Xception İçin Neleri Değiştirdik ve Neden Önemli?
Dinamik Faz Yönetimi (phase): phase="train", phase="val" parametreleri sayesinde kod, katmanlı mimariye tam uyum sağladı. Eğitim kodu train klasörüne bakarken, test kodu test klasörüne bakacak.

Xception Normalizasyonu: EfficientNet standart ImageNet normalizasyonu (mean=[0.485...]) kullanırken, Xception mimarisi verileri [-1, 1] aralığına sıkıştıran mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] normalizasyonunu bekler. 
Bu detay, Xception'ın doğru yakınsaması (convergence) için kritik önem taşır.

Akıllı Veri Artırma Ayrımı: val ve test aşamalarında görüntüye gürültü eklemek veya döndürmek modelin gerçek performansını ölçmemizi engeller. Bu yüzden phase="train" değilse veri artırma adımlarını otomatik olarak kapatacak mantığı kurduk.
'''


import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, phase="train", transform=None):
        """
        Geliştirilmiş Esnek Yol Çözücü Katman.
        Klasör hiyerarşisi ne olursa olsun 'train', 'val' veya 'test' 
        kelimelerini içeren alt yollardaki tüm resimleri otomatik yakalar.
        """
        # data/processed klasörünün tam mutlak yolunu alıyoruz
        self.root_dir = os.path.abspath(os.path.normpath(root_dir))
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Belirlenen fazın (train/val/test) aranacağı alt klasör adı
        self.phase_keyword = phase.lower()

        # data/processed altındaki tüm her şeyi derinlemesine tarıyoruz
        for root, _, files in os.walk(self.root_dir):
            # Yolun içinde 'train', 'val' veya 'test' anahtar kelimesi geçiyor mu kontrol et
            if self.phase_keyword in root.lower():
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        full_path = os.path.abspath(os.path.normpath(os.path.join(root, file)))
                        
                        # Resmin boyutunun 0 KB'tan büyük olduğunu doğrula
                        if os.path.getsize(full_path) > 0:
                            # Sınıflandırma Etiketi: Yolun veya dosyanın içinde 'fake' geçiyorsa 1, yoksa 0
                            if 'fake' in root.lower() or 'fake' in file.lower():
                                label = 1
                            else:
                                label = 0
                                
                            self.image_paths.append(full_path)
                            self.labels.append(label)

        print(f"📊 [{phase.upper()}] Seti Hazır: {len(self.image_paths)} adet yüz görüntüsü sisteme yüklendi.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = cv2.imread(img_path)
            if image is None: 
                raise ValueError("Boş Görüntü")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            try:
                image = np.array(Image.open(img_path).convert("RGB"))
            except Exception:
                # Olası bir bozuk dosyada eğitimi çökertmemek için dummy siyah resim üret
                image = np.zeros((299, 299, 3), dtype=np.uint8)
            
        label = self.labels[idx]
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, torch.tensor(label, dtype=torch.long)


def get_transforms(img_size=299, phase="train"):
    if phase == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
            ], p=0.3),
            A.OneOf([
                A.GaussianBlur(p=0.5),
                A.GaussNoise(p=0.5),
            ], p=0.3),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])