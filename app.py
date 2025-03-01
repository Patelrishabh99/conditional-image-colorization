import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from PIL import Image, ImageEnhance, ImageDraw
from io import BytesIO  # For download functionality



@st.cache_resource
def load_colorization_model():
    model = load_model("model.h5", compile=False)  # FIX: Load without compiling optimizer
    return model


# Load the model
model = load_colorization_model()



def process_image(image):
    image = np.array(image.convert("L"))  # Convert to grayscale
    image = cv2.resize(image, (512, 512))  # Resize to match model input
    image = image.astype("float32") / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    return image


def colorize_image(image):
    input_image = process_image(image)
    colorized_output = model.predict(input_image)  # Get model output
    colorized_output = np.squeeze(colorized_output)  # Remove extra dimensions
    colorized_output = (colorized_output * 255).astype("uint8")  # Convert to uint8 format
    return Image.fromarray(colorized_output)



def apply_custom_colors(image, selected_area, selected_color):

    draw = ImageDraw.Draw(image)

    # Define region coordinates (modify for better selection)
    x1, y1, x2, y2 = selected_area

    # Convert hex color to RGB
    selected_rgb = tuple(int(selected_color[i:i + 2], 16) for i in (1, 3, 5))

    # Apply selected color in the chosen region
    draw.rectangle([x1, y1, x2, y2], fill=selected_rgb, outline="black")

    return image



st.title("🎨 Advanced Image Colorization")
st.write("Upload a grayscale image and get a colorized version with **custom color selection**.")

# Upload Image
uploaded_image = st.file_uploader("📁 Choose a grayscale image...", type=["png", "jpg", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    # User selects region for custom colorization
    st.write("🎯 **Select the area to apply custom colors (x1, y1, x2, y2):**")
    x1 = st.slider("X1:", 0, image.width, 0)
    y1 = st.slider("Y1:", 0, image.height, 0)
    x2 = st.slider("X2:", 0, image.width, image.width - 600)
    y2 = st.slider("Y2:", 0, image.height, image.height - 480)

    # Color Picker for user-defined colors
    st.write("🎨 **Pick a color for the selected region:**")
    selected_color = st.color_picker("Choose a color", "#00FF00")  # Default: Green

    if st.button("🎨 Colorize Image"):
        colorized_image = colorize_image(image)

        # Apply user-selected colors
        colorized_image = apply_custom_colors(colorized_image, (x1, y1, x2, y2), selected_color)

        st.image(colorized_image, caption="🌈 Colorized Image with Custom Colors", use_column_width=True)

        # Convert image to bytes for download
        img_bytes = BytesIO()
        colorized_image.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

        # Provide download option
        st.download_button(label="💾 Download Colorized Image",
                           data=img_bytes,
                           file_name="colorized_output.png",
                           mime="image/png")
