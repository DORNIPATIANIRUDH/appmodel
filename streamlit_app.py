import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Define CNN model
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')  # 4 output classes
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize model
model = create_model()

# Define class labels
CLASS_LABELS = ["Fresh", "Dried", "Not Dried", "Rotten"]

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match CNN input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Fruit & Vegetable Freshness Classifier")
st.write("Upload an image to check its condition.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("Processing...")
    
    # Preprocess image
    processed_image = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed_image)[0]  # Get probability for each class
    
    # Show results for all classes
    st.write("### Classification Results:")
    for i, label in enumerate(CLASS_LABELS):
        st.write(f"**{label}:** {predictions[i] * 100:.2f}%")
    
    # Determine final prediction based on highest probabilities
    sorted_indices = np.argsort(predictions)[::-1]
    final_prediction = f"{CLASS_LABELS[sorted_indices[0]]} and {CLASS_LABELS[sorted_indices[1]]}" if predictions[sorted_indices[1]] > 0.2 else CLASS_LABELS[sorted_indices[0]]
    
    st.write(f"\n**Final Prediction:** {final_prediction}")
