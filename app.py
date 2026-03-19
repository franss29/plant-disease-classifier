import streamlit as st
import tensorflow as tf
import numpy as np
import json
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="PlantDoc AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --green-dark:  #1a3a2a;
    --green-mid:   #2d6a4f;
    --green-light: #52b788;
    --cream:       #f8f4ec;
    --accent:      #e9c46a;
    --danger:      #e76f51;
    --text-dark:   #1a1a1a;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--cream);
    color: var(--text-dark);
}

h1, h2, h3 { font-family: 'DM Serif Display', serif; }

/* Header */
.hero-header {
    background: linear-gradient(135deg, var(--green-dark) 0%, var(--green-mid) 60%, var(--green-light) 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: "🌿";
    font-size: 8rem;
    position: absolute;
    right: 2rem; top: 50%;
    transform: translateY(-50%);
    opacity: 0.15;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: white;
    margin: 0 0 0.4rem 0;
    line-height: 1.1;
}
.hero-subtitle {
    color: rgba(255,255,255,0.75);
    font-size: 1rem;
    font-weight: 300;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    color: white;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    margin-top: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.05em;
}

/* Upload zone */
.upload-zone {
    border: 2px dashed var(--green-light);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    background: white;
    transition: all 0.3s;
}

/* Result card */
.result-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    border-left: 5px solid var(--green-light);
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    margin-bottom: 1rem;
}
.result-card.healthy { border-left-color: var(--green-light); }
.result-card.disease { border-left-color: var(--danger); }

.disease-name {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    margin: 0 0 0.3rem 0;
}
.confidence-label {
    color: #666;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Progress bar custom */
.conf-bar-wrap { margin: 0.4rem 0; }
.conf-bar-label {
    display: flex; justify-content: space-between;
    font-size: 0.82rem; color: #555; margin-bottom: 2px;
}
.conf-bar-bg {
    background: #eee; border-radius: 99px; height: 8px; overflow: hidden;
}
.conf-bar-fill {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, var(--green-mid), var(--green-light));
    transition: width 1s ease;
}
.conf-bar-fill.top { background: linear-gradient(90deg, var(--green-dark), var(--green-light)); }

/* Info pill */
.info-pill {
    display: inline-block;
    background: #e8f5e9;
    color: var(--green-dark);
    border-radius: 99px;
    padding: 0.25rem 0.9rem;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0.15rem;
}
.info-pill.red { background: #fce8e4; color: var(--danger); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--green-dark) !important;
}
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] .stSelectbox label { color: rgba(255,255,255,0.7) !important; }

/* Stats strip */
.stats-strip {
    display: flex; gap: 1rem; margin-bottom: 1.5rem;
}
.stat-box {
    flex: 1; background: white; border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    text-align: center;
}
.stat-num {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem; color: var(--green-mid);
    display: block;
}
.stat-lbl { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }

/* Footer */
.footer { text-align: center; color: #aaa; font-size: 0.78rem; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e0e0e0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD MODEL & CLASS NAMES
# ─────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model AI...")
def load_model():
    import json

    # Baca jumlah kelas
    with open("class_names.json") as f:
        n = len(json.load(f))

    # Bangun ulang arsitektur
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights=None
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(128, 128, 3))
    x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(n, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    # Load bobot dari file
    model.load_weights('plant_disease_weights.weights.h5')
    return model

@st.cache_data
def load_class_names():
    with open("class_names.json") as f:
        return json.load(f)

def parse_class_name(raw_name):
    """Pisahkan nama tanaman dan penyakit dari format 'Tomato___Late_blight'."""
    parts = raw_name.replace("___", "|").replace("_", " ").split("|")
    plant = parts[0].strip() if len(parts) > 0 else raw_name
    disease = parts[1].strip() if len(parts) > 1 else "Unknown"
    return plant, disease

def is_healthy(disease_name):
    return "healthy" in disease_name.lower()


# ─────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────
def make_gradcam(model, img_array, class_idx):
    """Generate Grad-CAM heatmap untuk visualisasi area fokus model."""
    try:
        # Cari layer konvolusi terakhir
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.Model):
                for sub in reversed(layer.layers):
                    if len(sub.output_shape) == 4:
                        last_conv = sub.name
                        break
            if last_conv:
                break

        if last_conv is None:
            return None

        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv).output if last_conv in [l.name for l in model.layers]
                     else [l for l in model.layers if isinstance(l, tf.keras.Model)][0].get_layer(last_conv).output,
                     model.output]
        )

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            loss = preds[:, class_idx]

        grads = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap
    except Exception:
        return None


def overlay_gradcam(original_img, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap ke gambar asli."""
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    colormap = cm.get_cmap("RdYlGn_r")
    heatmap_colored = (colormap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
    overlaid = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
    return overlaid


# ─────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────
def predict(model, class_names, img_pil, img_size=(128, 128)):
    img_rgb = np.array(img_pil.convert("RGB"))
    img_resized = cv2.resize(img_rgb, img_size)
    img_array = np.expand_dims(img_resized.astype("float32"), axis=0)

    preds = model.predict(img_array, verbose=0)[0]
    top5_idx = np.argsort(preds)[::-1][:5]

    return preds, top5_idx, img_rgb, img_resized, img_array


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌿 PlantDoc AI")
    st.markdown("---")
    st.markdown("**Tentang Aplikasi**")
    st.markdown(
        "Aplikasi deteksi penyakit tanaman berbasis "
        "Computer Vision menggunakan **MobileNetV2** + Transfer Learning."
    )
    st.markdown("---")
    st.markdown("**Model Info**")
    st.markdown("- Architecture: MobileNetV2")
    st.markdown("- Input size: 128×128 px")
    st.markdown("- Training: 2-fase (frozen + fine-tune)")
    st.markdown("---")
    show_gradcam = st.toggle("🔍 Tampilkan Grad-CAM", value=True)
    top_k = st.slider("Jumlah prediksi ditampilkan", 3, 5, 3)
    st.markdown("---")
    st.markdown("*Portfolio Project — Data Scientist*")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

# Hero Header
st.markdown("""
<div class="hero-header">
    <p class="hero-title">PlantDoc AI 🌿</p>
    <p class="hero-subtitle">Deteksi penyakit tanaman dari foto daun secara otomatis</p>
    <span class="hero-badge">MobileNetV2 · Transfer Learning · 38 Kelas</span>
</div>
""", unsafe_allow_html=True)

# Load resources
try:
    model = load_model()
    class_names = load_class_names()
    NUM_CLASSES = len(class_names)
    model_loaded = True
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.info("Pastikan file `plant_disease_model.keras` dan `class_names.json` ada di folder yang sama dengan `app.py`.")
    model_loaded = False

if model_loaded:
    # Stats strip
    st.markdown(f"""
    <div class="stats-strip">
        <div class="stat-box"><span class="stat-num">{NUM_CLASSES}</span><span class="stat-lbl">Kelas Penyakit</span></div>
        <div class="stat-box"><span class="stat-num">MV2</span><span class="stat-lbl">Arsitektur</span></div>
        <div class="stat-box"><span class="stat-num">128px</span><span class="stat-lbl">Input Size</span></div>
        <div class="stat-box"><span class="stat-num">2-fase</span><span class="stat-lbl">Training</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Upload
    st.markdown("### 📷 Upload Foto Daun")
    uploaded = st.file_uploader(
        "Pilih gambar daun tanaman (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded:
        img_pil = Image.open(uploaded)
        preds, top5_idx, img_rgb, img_resized, img_array = predict(model, class_names, img_pil)

        top_idx = top5_idx[0]
        top_conf = preds[top_idx]
        plant, disease = parse_class_name(class_names[top_idx])
        healthy = is_healthy(disease)
        card_class = "healthy" if healthy else "disease"
        status_emoji = "✅" if healthy else "⚠️"

        col1, col2 = st.columns([1, 1.3], gap="large")

        # ── Kolom kiri: gambar
        with col1:
            st.markdown("**Gambar Input**")
            st.image(img_pil, use_container_width=True, caption="Foto yang diupload")

            if show_gradcam:
                heatmap = make_gradcam(model, img_array, top_idx)
                if heatmap is not None:
                    st.markdown("**Grad-CAM — Area Fokus Model**")
                    overlay = overlay_gradcam(img_resized, heatmap)
                    st.image(overlay, use_container_width=True,
                             caption="Merah = area paling diperhatikan model")
                else:
                    st.caption("Grad-CAM tidak tersedia untuk model ini.")

        # ── Kolom kanan: hasil
        with col2:
            st.markdown("**Hasil Diagnosis**")

            st.markdown(f"""
            <div class="result-card {card_class}">
                <p class="disease-name">{status_emoji} {disease}</p>
                <span class="info-pill">🌱 {plant}</span>
                {"<span class='info-pill'>Sehat</span>" if healthy else "<span class='info-pill red'>Terdeteksi Penyakit</span>"}
                <br><br>
                <span class="confidence-label">Tingkat Keyakinan Model</span>
                <div style="font-size:2rem;font-family:'DM Serif Display',serif;color:{'#2d6a4f' if healthy else '#e76f51'}">
                    {top_conf*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"**Top {top_k} Prediksi**")
            for rank, idx in enumerate(top5_idx[:top_k]):
                p_name = class_names[idx]
                conf = preds[idx] * 100
                _, dis = parse_class_name(p_name)
                bar_class = "top" if rank == 0 else ""
                st.markdown(f"""
                <div class="conf-bar-wrap">
                    <div class="conf-bar-label">
                        <span>{'🥇' if rank==0 else '🔹'} {p_name.replace('___',' → ').replace('_',' ')}</span>
                        <span><b>{conf:.1f}%</b></span>
                    </div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill {bar_class}" style="width:{conf:.1f}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Interpretasi
            st.markdown("<br>", unsafe_allow_html=True)
            if healthy:
                st.success("✅ Tanaman terlihat **sehat**. Tidak ditemukan tanda-tanda penyakit.")
            elif top_conf > 0.85:
                st.error(f"⚠️ Penyakit **{disease}** terdeteksi dengan keyakinan tinggi ({top_conf*100:.1f}%).")
            elif top_conf > 0.60:
                st.warning(f"🔶 Kemungkinan penyakit **{disease}**, namun keyakinan model sedang ({top_conf*100:.1f}%). Disarankan periksa ulang.")
            else:
                st.info(f"🔵 Prediksi kurang yakin ({top_conf*100:.1f}%). Coba foto dengan pencahayaan lebih baik.")

    else:
        # Placeholder saat belum ada gambar
        st.markdown("""
        <div class="upload-zone">
            <p style="font-size:3rem;margin:0">🍃</p>
            <p style="font-size:1.1rem;font-weight:600;color:#2d6a4f;margin:0.5rem 0">Upload foto daun tanaman</p>
            <p style="color:#888;font-size:0.85rem">Format JPG atau PNG · Pastikan daun terlihat jelas</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**💡 Tips untuk hasil terbaik:**")
        cols = st.columns(3)
        tips = [
            ("📸", "Foto close-up", "Pastikan daun mengisi sebagian besar frame"),
            ("💡", "Pencahayaan baik", "Gunakan cahaya alami, hindari bayangan"),
            ("🍃", "Daun tunggal", "Foto satu daun untuk hasil lebih akurat"),
        ]
        for col, (icon, title, desc) in zip(cols, tips):
            with col:
                st.markdown(f"""
                <div style="background:white;border-radius:12px;padding:1rem;text-align:center;box-shadow:0 2px 10px rgba(0,0,0,0.05)">
                    <div style="font-size:1.8rem">{icon}</div>
                    <div style="font-weight:600;font-size:0.9rem;margin:0.4rem 0">{title}</div>
                    <div style="color:#888;font-size:0.78rem">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    PlantDoc AI · Portfolio Project · Data Science & Machine Learning · MobileNetV2 Transfer Learning
</div>
""", unsafe_allow_html=True)