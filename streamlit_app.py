import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing import image

# Function to Load Pretrained Model
@st.cache_resource
def load_model():
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    # Freeze all layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification layers
    x = Flatten()(base_model.output)
    x = Dense(128, activation="relu")(x)
    x = Dense(3, activation="softmax")(x)  # 3 classes: Fresh, Rotten, Dried

    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model

# Load the model
model = load_model()

# Streamlit UI
st.title("Fruit & Vegetable Classifier 🍎🥦")
st.write("Upload an image of a **fruit or vegetable slice**, and the model will classify it as **Fresh, Rotten, or Dried**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Class Labels
labels = ["Fresh", "Rotten", "Dried"]

if uploaded_file is not None:
    # Convert image to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image to match MobileNetV2 input size
    img_resized = cv2.resize(img, (224, 224))

    # Preprocess image
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = preprocess_input(img_array)

    # Predict using the model
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    # Display Image and Prediction
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write(f"**Prediction:** {labels[class_idx]}")
    st.write(f"**Confidence:** {confidence:.2f}%")

