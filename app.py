import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os
from facenet_pytorch import MTCNN

# Katmanlı mimariden gerekli araçları çekiyoruz
from src.models import XceptionDetector
from src.utils import get_device

# Streamlit sayfa ayarları
st.set_page_config(page_title="Bozok Deepfake AI - Xception", layout="wide")

@st.cache_resource
def load_assets():
    """Model ve yüz tespiti kütüphanesini belleğe güvenli bir şekilde yükler."""
    device = get_device()
    
    # Xception modelini başlat ve en iyi ağırlıkları yükle
    model = XceptionDetector(pretrained=False)
    model_path = "checkpoints/xception1/best_xception_model.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        st.sidebar.success("✅ En iyi Xception ağırlıkları başarıyla yüklendi!")
    else:
        st.sidebar.warning("⚠️ Eğitilmiş model (best_xception_model.pth) bulunamadı! Rastgele ağırlıklar kullanılıyor.")
        
    model.to(device).eval()
    
    # Yüz tespiti için MTCNN kurulumu
    mtcnn = MTCNN(keep_all=False, device=device) # keep_all=False: Karedeki en baskın tek yüze odaklanır
    
    return model, mtcnn, device

# Varlıkları yükle
model, mtcnn, device = load_assets()

st.title("🛡️ Xception Tabanlı Deepfake Görüntü Analiz Paneli")
st.write("Bu panel, yüklenen videoların karelerini **Xception** mimarisi ve **MTCNN** yüz tespiti algoritması kullanarak gerçek zamanlı analiz eder.")

uploaded_file = st.file_uploader("Bir video dosyası seçin veya sürükleyin...", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Geçici bir dosyaya videoyu yazalım (OpenCV okuyabilsin diye)
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close() # Windows kilitlenmelerini önlemek için dosyayı kapatıyoruz
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📹 Kaynak Video")
        st.video(uploaded_file)
        
        if st.button("🚀 Analizi Başlat (20 Kare Analizi)"):
            video = cv2.VideoCapture(tfile.name)
            
            # Analiz listeleri
            probs = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Analiz Döngüsü
            while video.isOpened() and len(probs) < 20:
                ret, frame = video.read()
                if not ret:
                    break
                
                # BGR -> RGB Dönüşümü (MTCNN RGB bekler)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Kare üzerinde yüz tespiti yap
                face = mtcnn(frame_rgb)
                
                if face is not None:
                    # Yüzü Xception giriş boyutu olan 299x299'a getiriyoruz
                    face_input = torch.nn.functional.interpolate(face.unsqueeze(0), size=(299, 299)).to(device)
                    
                    # Xception normalizasyonunu uyguluyoruz [-1, 1] aralığı
                    face_input = (face_input - face_input.min()) / (face_input.max() - face_input.min()) # 0-1 aralığı
                    face_input = (face_input - 0.5) / 0.5 # [-1, 1] aralığı
                    
                    with torch.no_grad():
                        output = model(face_input)
                        prob = torch.sigmoid(output).item()
                        
                        # Olasılık değerini kaydet (1: Fake, 0: Real)
                        probs.append(prob)
                
                # İlerleme çubuğunu güncelle
                current_count = len(probs)
                progress_bar.progress(current_count / 20)
                status_text.text(f"İşlenen Geçerli Kare (Yüz Bulunan): {current_count}/20")
            
            video.release()
            os.unlink(tfile.name) # Geçici dosyayı diskten temizle
            
            # Verileri oturum hafızasına al
            st.session_state.probs = probs
            st.success("🎉 Analiz başarıyla tamamlandı!")

    # Sağ Sütunda Sadece Yüzdelik Sonucu Göster
    if 'probs' in st.session_state and len(st.session_state.probs) > 0:
        with col2:
            st.subheader("📊 Analiz Sonucu")
            
            # Tüm karelerin fake olasılıklarının ortalamasını alıyoruz
            average_fake_prob = np.mean(st.session_state.probs)
            
            fake_percentage = average_fake_prob * 100
            real_percentage = (1 - average_fake_prob) * 100
            
            # Kullanıcıya büyük ve net bir sonuç kartı gösterelim
            if fake_percentage > 50:
                st.error(f"🚨 Bu video büyük ihtimalle **SAHTE (DEEPFAKE)**!")
            else:
                st.success(f"✅ Bu video büyük ihtimalle **GERÇEK**.")
            
            # Yüzdelik göstergeler
            st.write("---")
            st.metric(label="Gerçeklik Oranı", value=f"%{real_percentage:.1f}")
            st.metric(label="Sahtelik (Deepfake) Oranı", value=f"%{fake_percentage:.1f}")
            
            # Streamlit ilerleme çubuğu ile görsel destek
            st.write("**Olasılık Dağılımı:**")
            st.progress(int(fake_percentage))
            st.caption(f"Sol taraf %0 (Tamamen Gerçek) ———— Sağ taraf %100 (Tamamen Fake)")