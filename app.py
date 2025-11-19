# app.py (model-loading + downloader snippet)
import os
import streamlit as st
import tensorflow as tf
import requests
import shutil
import traceback
import numpy as np

st.write("Python:", os.sys.version.splitlines()[0])
st.write("TensorFlow:", tf.__version__)

MODEL_LOCAL_KERAS = "model.keras"
MODEL_LOCAL_H5 = "model.h5"

# --- CONFIG: put your direct-download URL or Google Drive file id here ---
# Option 1: direct URL (raw GitHub, S3, file host) - recommended if file < GitHub limit
MODEL_DOWNLOAD_URL = st.secrets.get("MODEL_URL", None)  # set via Streamlit secrets or replace with string

# Option 2: Google Drive file id (if you upload there)
# Example: if share link is https://drive.google.com/file/d/FILEID/view?usp=sharing -> FILEID is the ID
GDRIVE_FILE_ID = st.secrets.get("GDRIVE_FILE_ID", None)

# Helper: download from generic URL
def download_from_url(url, out_path):
    st.info(f"Downloading model from URL: {url}")
    try:
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        st.success(f"Downloaded to {out_path}")
        return True
    except Exception as e:
        st.warning(f"Download failed: {e}")
        st.text(traceback.format_exc())
        return False

# Helper: simple Google Drive downloader for files < ~100MB
def download_from_gdrive(file_id, out_path):
    # This works for files that don't require a confirm token (small files)
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    return download_from_url(url, out_path)

# Create fallback model (same architecture you gave)
def create_new_model(input_dim=11, save_path=None):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    if save_path:
        try:
            model.save(save_path)
            st.info(f"Saved fallback model to {save_path}")
        except Exception as e:
            st.warning(f"Could not save fallback model: {e}")
    return model

# Try to ensure a model file exists locally
def ensure_model_present():
    # If present already, nothing to do
    if os.path.exists(MODEL_LOCAL_KERAS) or os.path.exists(MODEL_LOCAL_H5):
        return True

    # Try direct URL if configured
    if MODEL_DOWNLOAD_URL:
        # choose filename based on url
        fname = MODEL_LOCAL_H5 if MODEL_DOWNLOAD_URL.endswith(".h5") else MODEL_LOCAL_KERAS
        ok = download_from_url(MODEL_DOWNLOAD_URL, fname)
        if ok:
            return True

    # Try Google Drive if configured
    if GDRIVE_FILE_ID:
        # attempt .h5 first then .keras
        for fname in (MODEL_LOCAL_H5, MODEL_LOCAL_KERAS):
            ok = download_from_gdrive(GDRIVE_FILE_ID, fname)
            if ok:
                return True

    return False

# Load model with fallbacks
def load_model_with_fallback():
    # Ensure model file exists (try downloads)
    ensure_model_present()

    # 1) prefer model.keras
    try:
        if os.path.exists(MODEL_LOCAL_KERAS):
            st.info("Loading model.keras ...")
            model = tf.keras.models.load_model(MODEL_LOCAL_KERAS)
            st.success("Loaded model.keras")
            return model
    except Exception as e:
        st.warning(f"Failed to load model.keras: {e}")
        st.text(traceback.format_exc())

    # 2) try model.h5 with compile=False, then recompile
    try:
        if os.path.exists(MODEL_LOCAL_H5):
            st.info("Loading model.h5 with compile=False ...")
            os.environ['TF_USE_LEGACY_KERAS'] = '1'
            model = tf.keras.models.load_model(MODEL_LOCAL_H5, compile=False)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            st.success("Loaded model.h5 and recompiled")
            # Optionally, save as new format if possible
            try:
                model.save(MODEL_LOCAL_KERAS)
                st.info("Saved model as model.keras")
            except Exception:
                st.warning("Could not save model.keras (maybe large file or FS restrictions).")
            return model
    except Exception as e:
        st.warning(f"Failed to load model.h5: {e}")
        st.text(traceback.format_exc())

    # 3) final fallback: create new model with random weights
    st.error("No usable model found. Creating fallback model with random weights.")
    model = create_new_model(input_dim=11, save_path=None)  # not saving by default
    return model

# Main: load
model = load_model_with_fallback()

# Example prediction UI so app doesn't crash
st.header("Quick test prediction")
if st.button("Run test prediction"):
    x = np.random.rand(1, 11).astype(np.float32)
    try:
        pred = model.predict(x, verbose=0)
        st.write("Prediction:", float(pred.squeeze()))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.text(traceback.format_exc())
