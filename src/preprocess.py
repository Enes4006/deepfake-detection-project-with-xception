import os
import cv2
import torch
from tqdm import tqdm
from facenet_pytorch import MTCNN
from src.utils import get_device

def extract_faces(raw_dir="data/raw", processed_dir="data/processed", frame_skip=30): # 10 idi
    """
    raw_dir altındaki videoları okur, yüzleri kırpar ve processed_dir altına kaydeder.
    frame_skip: Her X karede bir işlem yap (Video akışını hızlandırmak ve benzer kareleri elemek için).
    """
    device = get_device()
    mtcnn = MTCNN(keep_all=False, device=device)
    
    categories = ['Real', 'Fake']
    
    print("✂️ Videolardan yüz ayıklama işlemi başlatıldı...")
    
    for cat in categories:
        cat_raw_path = os.path.join(raw_dir, cat)
        if not os.path.exists(cat_raw_path):
            continue
            
        videos = [v for v in os.listdir(cat_raw_path) if v.endswith(('.mp4', '.avi', '.mov'))]
        
        for video_name in tqdm(videos, desc=f"İşlenen Sınıf: {cat}"):
            video_path = os.path.join(cat_raw_path, video_name)
            video_id = os.path.splitext(video_name)[0]
            
            # Her video için processed altında benzersiz bir klasör oluştur
            output_video_dir = os.path.join(processed_dir, cat, video_id)
            os.makedirs(output_video_dir, exist_ok=True)
            
            cap = cv2.VideoCapture(video_path)
            frame_idx = 0
            saved_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_idx % frame_skip == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Yüzü tespit et ve kırpılmış boyut sınırlarını al
                    boxes, _ = mtcnn.detect(frame_rgb)
                    
                    if boxes is not None:
                        box = boxes[0].astype(int)
                        # Sınırların dışına taşmayı önle
                        img_h, img_w, _ = frame.shape
                        x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(img_w, box[2]), min(img_h, box[3])
                        
                        face_img = frame[y1:y2, x1:x2]
                        
                        if face_img.size > 0:
                            out_path = os.path.join(output_video_dir, f"frame_{frame_idx}.jpg")
                            cv2.imwrite(out_path, face_img)
                            saved_count += 1
                            
                frame_idx += 1
            cap.release()
            
    print("✅ Yüz ayıklama operasyonu başarıyla tamamlandı!")

if __name__ == "__main__":
    extract_faces()