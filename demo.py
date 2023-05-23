import streamlit as st
import keras
import os
import cv2
import numpy as np

from keras import layers
from cnn_model import cnn


model = cnn((256, 256, 3), 3)
model.load_weights("model_weights.h5")

imgs = []
labels = []
for path in os.listdir("test_imgs"):
    if path.startswith("cardboard") or path.startswith("glass") or path.startswith("metal"):
        continue

    img = cv2.imread(os.path.join("test_imgs", path))
    resized = cv2.resize(img, (256, 256))
    labels.append(path.split("_")[0])
    imgs.append(resized)


if "chosen" not in st.session_state.keys():
    st.session_state["chosen"] = imgs[0]
    st.session_state["label"] = labels[0]
    st.session_state["text_1"] = "Not predicted"
    st.session_state["text_2"] = ""


def gen_rand_img():
    rand_idx = np.random.randint(len(imgs))
    st.session_state["chosen"] = imgs[rand_idx]
    st.session_state["label"] = labels[rand_idx]


def predict_img():
    classes = ["paper", "plastic", "trash"]
    
    with st.spinner("Predicting image"):
        X = st.session_state["chosen"]
        y = st.session_state["label"]
        prob = model.predict(np.asarray([X]))
        pred = np.argmax(prob, axis = 1)[0]
        st.session_state["text_1"] = f"Predicted Label: {classes[pred]}"
        st.session_state["text_2"] = f"True Label: {y}"
        

st.title("Model demo")
left, right = st.columns(2)
left.image(st.session_state["chosen"])
right.text(st.session_state["text_1"])
right.text(st.session_state["text_2"])

left, right = st.columns(2)
left.button("Random Image", on_click = gen_rand_img)
right.button("Predict", on_click = predict_img)