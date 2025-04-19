import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pre-trained model
model = load_model("brain_tumor_classifier.h5")

# Class labels (same as used in training)
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Define image size (same as used in model training)
IMG_SIZE = (224, 224)

# Image augmentation setup
datagen = ImageDataGenerator(
    rescale=1./255,
    brightness_range=(0.8, 1.2),
    rotation_range=20,
    zoom_range=0.15,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Function to process and predict the image
def predict_image(image):
    img = Image.open(image)
    img = img.resize(IMG_SIZE)

    img = img.convert('RGB')

    img_array = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Augment the image using ImageDataGenerator
    augmented_image = next(datagen.flow(img_array))  # Apply augmentation

    # Make prediction with the augmented image
    predictions = model.predict(augmented_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)  # Get the confidence of the prediction
    return class_labels[predicted_class], confidence, predictions

# Streamlit UI setup
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="centered")
st.title("🧠 Brain Tumor Classifier")

# Add a description for the web app
st.write("""
Upload an MRI image of the brain below, and the classifier will predict the type of brain tumor.
We use a pre-trained deep learning model to detect gliomas, meningiomas, pituitary tumors, and healthy brain scans (no tumor).
""")

# Image upload section
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image preview with a shadow effect
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True, output_format='PNG')

    # Prediction when the user clicks the button
    if st.button("Predict"):
        # Call the prediction function
        prediction, confidence, predictions = predict_image(uploaded_file)
        
        # Display the prediction result in a more detailed way
        st.subheader("Prediction Result")
        st.write(f"### Predicted Class: **{prediction}**")
        st.write(f"### Confidence: **{confidence*100:.2f}%**")

        # Provide some more details about each class
        st.write("### Detailed Explanation:")
        if prediction == 'Glioma':
            st.write("""
            **Gliomas** are a type of brain tumor that begins in the glial cells. They are the most common type of primary brain tumor.
            Gliomas can be benign (non-cancerous) or malignant (cancerous).
            """)
        elif prediction == 'Meningioma':
            st.write("""
            **Meningiomas** are tumors that form in the meninges, the layers of tissue covering the brain and spinal cord.
            They are usually benign, but some may be malignant.
            """)
        elif prediction == 'No Tumor':
            st.write("""
            **No Tumor**: The MRI scan appears to be normal, with no signs of a tumor detected.
            However, it is always advised to consult a healthcare professional for a full diagnosis.
            """)
        elif prediction == 'Pituitary':
            st.write("""
            **Pituitary Tumors** are abnormal growths that develop in the pituitary gland located at the base of the brain.
            These can affect hormone production and lead to various symptoms.
            """)

        # Confidence bar with a visually appealing plot
        st.write("### Confidence Bar:")
        fig, ax = plt.subplots(figsize=(6, 1))
        ax.barh([0], confidence, color='green', edgecolor='black')
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_title(f"Confidence: {confidence*100:.2f}%")
        st.pyplot(fig)

        # Plot model's confidence for each class
        st.write("### Model Confidence for Each Class:")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=class_labels, y=np.squeeze(predictions), palette="coolwarm", ax=ax)
        ax.set_title("Class Prediction Probabilities")
        ax.set_ylabel("Confidence")
        ax.set_xlabel("Classes")
        st.pyplot(fig)

        # Add some styling to the output
        st.markdown(
            f"""
            <style>
                .stProgress > div > div {{
                    background-color: {'#ff4d4d' if confidence < 0.5 else '#66cc66'};
                }}
            </style>
            """, unsafe_allow_html=True)

# Additional styling for the UI
st.markdown("""
    <style>
        .title {
            font-size: 30px;
            font-weight: bold;
        }
        .description {
            font-size: 16px;
            color: gray;
        }
        img {
            box-shadow: 0px 0px 20px rgba(0,0,0,0.3);
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)
