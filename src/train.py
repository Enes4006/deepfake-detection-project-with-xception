import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import roc_auc_score

# Katmanlı mimari importları
from src.dataset import DeepfakeDataset, get_transforms
from src.models import XceptionDetector
from src.utils import get_device

def run_validation(model, val_loader, criterion, device):
    """
    Modelin eğitim sırasında görmediği doğrulama (validation) setindeki 
    performansını hesaplar (Overfitting kontrolü için kritik katman).
    """
    model.eval() # Modeli değerlendirme moduna al (Dropout'u kapatır)
    val_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad(): # Gradyan hesaplamayı kapatarak bellek tasarrufu sağlar
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().view(-1, 1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = val_loss / len(val_loader)
    try:
        avg_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        avg_auc = 0.5 # Eğer veri setinde tek bir sınıfa ait kareler kaldıysa çökmesin
        
    return avg_loss, avg_auc

def train_model():
    # 1. Konfigürasyon Ayarları
    device = get_device()
    lr = 1e-5
    batch_size = 8  # GTX 1650 (4GB VRAM) ve 299x299 çözünürlük için en güvenli sınır = 4
    epochs = 5
    img_size = 299  # Xception için zorunlu boyut
    
    torch.cuda.empty_cache() # GPU belleğini sıfırla

    # 2. Veri Yükleyicileri (Train ve Val Ayrı Kanallardan Yüklenir)
    train_dataset = DeepfakeDataset(root_dir="data/processed", phase="train", transform=get_transforms(img_size, phase="train"))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) # num_workers=4 ile veri yükleme işlemini hızlandırır, pin_memory=True ile CPU-GPU transferini optimize eder
    
    val_dataset = DeepfakeDataset(root_dir="data/processed", phase="val", transform=get_transforms(img_size, phase="val"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) # num_workers=4 ile veri yükleme işlemini hızlandırır, pin_memory=True ile CPU-GPU transferini optimize eder
    
    # 3. Model, Kayıp Fonksiyonu ve Optimizer Kurulumu
    model = XceptionDetector(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-8)
    scaler = torch.amp.GradScaler('cuda') # VRAM tasarrufu için Mixed Precision

    # Checkpoint klasörünü ayarla
    os.makedirs("checkpoints/xception1", exist_ok=True)

    print(f"🚀 Xception Mimarisi Eğitimi Başlıyor... Cihaz: {device}")
    print(f"📈 Eğitim Örneği: {len(train_dataset)} | Doğrulama Örneği: {len(val_dataset)}")
    
    best_val_auc = 0.0 # En iyi modeli takip etmek için referans skor

    # 4. Eğitim Döngüsü (Training Loop)
    for epoch in range(epochs):
        model.train() # Modeli eğitim moduna al (Dropout ve BatchNorm aktif)
        running_loss = 0.0
        all_labels = []
        all_preds = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().view(-1, 1)
            
            optimizer.zero_grad()
            
            # Karışık Hassasiyetli İleri Besleme
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if torch.isnan(loss): 
                continue
            
            # Geriye Yayılım ve Gradyan Kırpma
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'train_loss': f'{loss.item():.4f}'})

        # Epoch Sonu Performans Değerlendirmeleri
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_auc = roc_auc_score(all_labels, all_preds)
        
        # Validation Adımını Tetikle
        epoch_val_loss, epoch_val_auc = run_validation(model, val_loader, criterion, device)
        
        print(f"\n📊 Epoch {epoch+1} Özet:")
        print(f"   Eğitim    -> Loss: {epoch_train_loss:.4f} | AUC: {epoch_train_auc:.4f}")
        print(f"   Doğrulama -> Loss: {epoch_val_loss:.4f} | AUC: {epoch_val_auc:.4f}")
        
        # 5. En İyi Modeli Kaydetme Mantığı (Model Checkpointing)
        if epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            torch.save(model.state_dict(), "checkpoints/xception1/best_xception_model.pth")
            print(f"🌟 En iyi model güncellendi! Yeni Best Val AUC: {best_val_auc:.4f}")
            
        # Her ihtimale karşı son epoch ağırlıklarını da yedekle
        torch.save(model.state_dict(), f"checkpoints/xception1/xception_epoch_{epoch+1}.pth")
        print("-" * 50)

if __name__ == "__main__":
    train_model()






'''
Bu yeni eğitim dosyasında şunları yapacağız:
train veri yükleyicisi ile modeli eğiteceğiz.
Her epoch sonunda val (doğrulama) veri yükleyicisi ile modelin hiç görmediği verideki Validation Loss ve Validation AUC değerlerini hesaplayacağız.
Eğer o epoch'taki doğrulama başarısı eskisinden iyiyse, en iyi modeli (best_model.pth) kaydederek Erken Durdurma (Early Stopping) mantığına zemin hazırlayacağız.


run_validation Fonksiyonu: Eğitim döngüsünün dışına izole bir doğrulama katmanı kurduk. Modelin ağırlıklarını güncellemeden, sadece saf performans ölçümü yapar. 
Modelin ezberlemesini engellemek için her epoch sonunda validation kaybını takip ettim
clip_grad_norm_ ve GradScaler: Xception gibi derin ağların 4GB kartta hata vermeden, kararlı ve sarsıntısız eğitilmesini sağlayan görünmez kahramanlarımız burada da görev başında.
'''

