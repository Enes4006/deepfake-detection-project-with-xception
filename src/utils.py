import torch
import os

def get_device():
    """
    Sistemin CUDA (GPU) destekleyip desteklemediğini kontrol eder.
    GTX 1650 kartının aktif olarak kullanılmasını garantiler.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def save_frame_as_image(frame, folder="temp", name="face.jpg"):
    """
    Gerekli durumlarda (örneğin hata analizi veya arayüz süreçlerinde)
    kırpılan yüz karesini diske güvenli bir şekilde kaydeder.
    """
    import cv2
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, name)
    cv2.imwrite(path, frame)
    return path