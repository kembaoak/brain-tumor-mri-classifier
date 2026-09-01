import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "model.weights.h5")
EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "examples")
IMG_SIZE = (224, 224)
NUM_CLASSES = 4

# Alphabetical order used by tf.keras.utils.image_dataset_from_directory during training.
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}

DISCLAIMER = (
    "**Disclaimer:** This project is intended for educational and research purposes only. "
    "The resulting model should not be used as a medical diagnostic tool."
)


@st.cache_resource
def load_model():
    # Mirrors the architecture built in MRI_Image_Recognition.ipynb exactly, so the
    # saved weights (which don't carry Keras-version-specific config serialization,
    # unlike a full .keras file) load in cleanly regardless of Keras patch version.
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights=None,
    )

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.load_weights(WEIGHTS_PATH)
    return model


def predict(image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE)
    array = np.array(image).astype("float32")
    array = np.expand_dims(array, axis=0)

    # The model's graph already applies mobilenet_v2.preprocess_input internally,
    # so raw 0-255 pixel values must be passed in here (no manual normalization).
    probs = load_model().predict(array, verbose=0)[0]

    return {name: float(prob) for name, prob in zip(CLASS_NAMES, probs)}


st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    footer { visibility: hidden; }
    img { border-radius: 10px; }
    div[data-testid="stFileUploaderDropzone"] { border-radius: 12px; }
    .hero-title { text-align: center; font-size: 2.2rem; font-weight: 700; margin-bottom: 0; }
    .hero-subtitle { text-align: center; color: var(--text-color, #8a8a8a); font-size: 1.05rem; margin-top: 0.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🧠 About")
    st.markdown(
        "This app classifies brain MRI scans into four categories using a "
        "**MobileNetV2** convolutional network fine-tuned via transfer learning."
    )
    st.markdown("**Classes**")
    st.markdown("- Glioma\n- Meningioma\n- Pituitary tumor\n- No tumor")
    st.metric("Test accuracy", "~83%")
    st.markdown(
        "**Dataset:** [Brain Tumor MRI Dataset on Kaggle]"
        "(https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)"
    )
    st.divider()
    st.markdown(f":red[{DISCLAIMER}]")

st.markdown('<p class="hero-title">🧠 Brain Tumor MRI Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Upload a brain MRI scan and get an instant prediction with confidence scores.</p>',
    unsafe_allow_html=True,
)
st.write("")

if "selected_example" not in st.session_state:
    st.session_state.selected_example = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

with st.container(border=True):
    st.markdown("##### 📤 Upload a brain MRI scan")
    uploaded_file = st.file_uploader(
        "Upload a brain MRI scan",
        type=["jpg", "jpeg", "png"],
        key=f"uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )

    st.caption("...or try an example")
    example_files = sorted(os.listdir(EXAMPLE_DIR))
    cols = st.columns(len(example_files))
    for col, fname in zip(cols, example_files):
        path = os.path.join(EXAMPLE_DIR, fname)
        label = os.path.splitext(fname)[0]
        with col:
            st.image(path, caption=CLASS_LABELS.get(label, label), use_container_width=True)
            if st.button("Use this", key=fname, use_container_width=True):
                st.session_state.selected_example = path
                # Force the uploader to remount empty, so it visually reflects
                # that the example has replaced whatever was previously uploaded.
                st.session_state.uploader_key += 1
                st.rerun()

image = None
if uploaded_file is not None:
    st.session_state.selected_example = None
    image = Image.open(uploaded_file)
elif st.session_state.selected_example:
    image = Image.open(st.session_state.selected_example)

if image is not None:
    st.write("")
    with st.container(border=True):
        st.markdown("##### 🔍 Result")
        col_img, col_result = st.columns([1, 1.3], gap="large")

        with col_img:
            st.image(image, use_container_width=True)

        with col_result:
            with st.spinner("Classifying..."):
                result = predict(image)

            sorted_result = dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))
            top_class, top_conf = next(iter(sorted_result.items()))

            st.metric("Prediction", CLASS_LABELS[top_class], f"{top_conf:.1%} confidence")
            st.markdown("**Confidence breakdown**")
            for cls, prob in sorted_result.items():
                st.caption(f"{CLASS_LABELS[cls]} — {prob:.1%}")
                st.progress(prob)
