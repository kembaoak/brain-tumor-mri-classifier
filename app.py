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


st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠")
st.title("🧠 Brain Tumor MRI Classifier")
st.markdown(
    "Upload a brain MRI scan to classify it as **glioma**, **meningioma**, "
    "**pituitary tumor**, or **no tumor**, using a MobileNetV2 model fine-tuned "
    "on the Brain Tumor MRI Dataset."
)
st.markdown(f":red[{DISCLAIMER}]")

if "selected_example" not in st.session_state:
    st.session_state.selected_example = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

st.subheader("Upload a brain MRI scan")
uploaded_file = st.file_uploader(
    "Upload a brain MRI scan",
    type=["jpg", "jpeg", "png"],
    key=f"uploader_{st.session_state.uploader_key}",
    label_visibility="collapsed",
)

st.caption("...or try an example:")
example_files = sorted(os.listdir(EXAMPLE_DIR))
cols = st.columns(len(example_files))
for col, fname in zip(cols, example_files):
    path = os.path.join(EXAMPLE_DIR, fname)
    label = os.path.splitext(fname)[0]
    with col:
        st.image(path, caption=CLASS_LABELS.get(label, label))
        if st.button("Use this", key=fname):
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
    st.subheader("Selected image")
    st.image(image, width=300)
    with st.spinner("Classifying..."):
        result = predict(image)

    sorted_result = dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))
    top_class, top_conf = next(iter(sorted_result.items()))
    st.success(f"Prediction: **{CLASS_LABELS[top_class]}** ({top_conf:.1%} confidence)")
    st.bar_chart({CLASS_LABELS[k]: v for k, v in sorted_result.items()})
