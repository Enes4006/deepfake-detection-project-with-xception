import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def calculate_metrics(all_labels, all_preds_binary, all_probs):
    """
    Eğitim, arayüz (Streamlit) ve Jupyter Notebook test süreçlerinde
    kullanılan tüm akademik metrikleri hesaplar.
    """
    precision = precision_score(all_labels, all_preds_binary, zero_division=0)
    recall = recall_score(all_labels, all_preds_binary, zero_division=0)
    f1 = f1_score(all_labels, all_preds_binary, zero_division=0)
    
    # AUC (Eğri Altındaki Alan) Hesabı
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_value = auc(fpr, tpr)
    
    return {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUC": auc_value
    }

def plot_confusion_matrix(all_labels, all_preds_binary):
    """
    Matplotlib ve Seaborn kullanarak standart bir Karmaşıklık Matrisi üretir.
    Streamlit (st.pyplot) ve Jupyter Notebook ile tam uyumludur.
    """
    cm = confusion_matrix(all_labels, all_preds_binary, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Gerçek', 'Sahte'], 
                yticklabels=['Gerçek', 'Sahte'], ax=ax)
    
    ax.set_ylabel('Gerçek Etiket')
    ax.set_xlabel('Model Tahmini')
    ax.set_title('Karmaşıklık Matrisi (Confusion Matrix)')
    plt.tight_layout()
    return fig


def get_gradcam_visualization(model, input_tensor, original_frame):
    """
    Açıklanabilir Yapay Zeka (XAI): Xception modelinin deepfake kararı verirken 
    yüz üzerinde odaklandığı alanları (ısı haritası olarak) hesaplar.
    """
    import torch.nn as nn

    # Versiyon farklarından etkilenmemek için model omurgasındaki en son Conv2d katmanını dinamik olarak buluyoruz
    target_layers = []
    for module in reversed(list(model.backbone.modules())): # yapısı sayesinde kod, Xception ağını sondan başa doğru tarar ve 
                                                            #karşısına çıkan ilk 2B evrişim (Conv2d) katmanını otomatik olarak Grad-CAM hedefi seçer.
        if isinstance(module, nn.Conv2d):
            target_layers = [module]
            break

    # Eğer şans eseri hiçbir Conv2d bulunamazsa (güvenlik önlemi) eski sisteme fallback yap
    if not target_layers:
        try:
            target_layers = [model.backbone.conv_head]
        except AttributeError:
            # Alternatif katman adı
            target_layers = [model.backbone.act2]
    
    # Grad-CAM nesnesini oluştur
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # İkili sınıflandırma çıktısını hedef al (0. indeksteki logit skor)
    targets = [ClassifierOutputTarget(0)]
    
    # Isı haritası (Siyah-Beyaz gradyan) maskesini üret
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    # Orijinal görüntüyü normalize et (0-1 arası float matrisi)
    img_float = original_frame.astype(np.float32) / 255.0
    img_float = cv2.resize(img_float, (299, 299)) # Xception boyut eşleşmesi
    
    # Isı haritasını orijinal yüz görüntüsünün üzerine bindir (RGB olarak)
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    return visualization

'''

def get_gradcam_visualization(model, input_tensor, original_frame):
    """
    Açıklanabilir Yapay Zeka (XAI): Xception modelinin deepfake kararı verirken 
    yüz üzerinde odaklandığı alanları (ısı haritası olarak) hesaplar.
    """
    # Xception mimarisinin en son evrişimsel (feature extraction) katmanını hedefliyoruz
    target_layers = [model.backbone.conv_head]
    
    # Grad-CAM nesnesini oluştur
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # İkili sınıflandırma çıktısını hedef al (0. indeksteki logit skor)
    targets = [ClassifierOutputTarget(0)]
    
    # Isı haritası (Siyah-Beyaz gradyan) maskesini üret
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    # Orijinal görüntüyü normalize et (0-1 arası float matrisi)
    img_float = original_frame.astype(np.float32) / 255.0
    img_float = cv2.resize(img_float, (299, 299)) # Xception boyut eşleşmesi
    
    # Isı haritasını orijinal yüz görüntüsünün üzerine bindir (RGB olarak)
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    return visualization
'''
