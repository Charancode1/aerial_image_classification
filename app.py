# app.py — FAKE demo Streamlit app for Bird vs Drone (deterministic fake predictions + fake Grad-CAM)
import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import io, hashlib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Aerial Classifier (Demo)", layout="centered")
st.title("Aerial Object Classifier — Bird vs Drone (Demo)")


IMG_SIZE = 224
CLASS_NAMES = ["bird", "drone"]

def deterministic_probs_from_image(img_bytes):
    """
    Deterministic pseudo-probability generator:
    - Hash the image bytes to produce stable results.
    - Use hash to bias result + a small influence from brightness.
    """
    h = int(hashlib.sha256(img_bytes).hexdigest()[:8], 16)
    base = (h % 1000) / 1000.0  # [0,1)
    # use brightness as a minor signal
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((64,64))
    arr = np.array(img) / 255.0
    brightness = arr.mean()  # 0..1
    # combine deterministically
    drone_prob = 0.25 * brightness + 0.75 * base
    drone_prob = max(0.0, min(1.0, drone_prob))
    bird_prob = 1.0 - drone_prob
    return {"bird": bird_prob, "drone": drone_prob}, h

def make_fake_heatmap(img, seed):
    """
    Create a fake Grad-CAM-like heatmap (gaussian blob) using the seed.
    Returns RGBA image to overlay.
    """
    np.random.seed(seed & 0xFFFF)
    w, h = img.size
    # create coordinate grid
    xx, yy = np.meshgrid(np.linspace(0,1,w), np.linspace(0,1,h))
    # seed determines center and spread
    cx = (seed % 1000) / 1000.0
    cy = ((seed >> 8) % 1000) / 1000.0
    sigma = 0.08 + ((seed >> 16) % 50) / 200.0
    blob = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2*sigma*sigma))
    # add a second small random blob to look more realistic
    cx2 = ((seed >> 5) % 1000) / 1000.0
    cy2 = ((seed >> 13) % 1000) / 1000.0
    blob += 0.6 * np.exp(-((xx - cx2)**2 + (yy - cy2)**2) / (2*(sigma*1.5)**2))
    blob = (blob - blob.min()) / (blob.max() - blob.min() + 1e-9)
    # convert to heatmap RGBA
    cmap = plt.get_cmap("jet")
    rgba = cmap(blob)  # HxWx4
    # convert to PIL Image and resize to original
    heat = Image.fromarray((rgba * 255).astype(np.uint8)).resize((w,h))
    return heat

# File uploader
uploaded = st.file_uploader("Upload an aerial image (jpg/png)", type=["jpg","jpeg","png"])
col1, col2 = st.columns([1,1])

if uploaded is None:
    st.info("Upload an image to see a fake prediction and Grad-CAM overlay.")
else:
    img_bytes = uploaded.read()
    uploaded.seek(0)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    # show original on left
    with col1:
        st.image(img, caption="Input image", use_column_width=True)

    # compute deterministic fake probs
    probs, seed = deterministic_probs_from_image(img_bytes)
    pred_label = max(probs, key=probs.get)
    pred_prob = probs[pred_label]

    # show prediction and bar chart on right
    with col2:
        st.markdown(f"### Prediction: **{pred_label.upper()}**")
        st.markdown(f"**Confidence:** {pred_prob:.3f}")
        st.bar_chart({"bird": float(probs["bird"]), "drone": float(probs["drone"])})

    # create and show fake Grad-CAM overlay
    heat = make_fake_heatmap(img, seed)
    # blend overlay
    overlay_alpha = 0.45
    blended = Image.blend(img.convert("RGBA"), heat.convert("RGBA"), alpha=overlay_alpha)
    st.image(blended, caption="Fake Grad-CAM overlay (demo)", use_column_width=True)

    # show some meta info for demo
    st.markdown("---")
    st.write("Demo info:")
    st.write(f"- Deterministic seed (hex): `{hex(seed)}`")
    avg_brightness = np.array(img).mean() / 255.0
    st.write(f"- Avg brightness (used in fake logic): {avg_brightness:.3f}")

st.write("---")

