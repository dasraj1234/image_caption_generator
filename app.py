# app.py

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
import base64
import os
from io import BytesIO

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Create OpenAI client
client = OpenAI(api_key=api_key)

# ----------- Function to Convert Image to Base64 ----------- #
def encode_image(image: Image.Image) -> str:
    """
    Converts a PIL Image to base64 encoded string.
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ----------- Function to Generate Caption ----------- #
def generate_caption(instruction: str, image_base64: str, model="gpt-4-turbo") -> str:
    """
    Sends image and instruction to GPT-4-turbo to generate a caption.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes captions for images based on the user's instructions."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail": "auto"}}
                    ]
                }
            ],
            max_tokens=300,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {e}"

# ----------- Streamlit UI ----------- #

# Page Configuration
st.set_page_config(page_title="🖼️ AI Image Caption Generator", layout="centered")

# Title
st.title("🧠 AI Image Caption Generator")

# Description
st.markdown("Upload an image and tell the AI how to caption it (e.g., 'Write a funny caption').")

# File Uploader
uploaded_file = st.file_uploader("Upload an image (jpeg, png, jpg)", type=["jpeg", "png", "jpg"])

# Instruction Input
instruction = st.text_input("Instruction (e.g., 'Write a poetic caption')", "")

# Spacer
st.write("")

# Button to Trigger Caption Generation
if st.button("✨ Generate Caption"):
    if not uploaded_file:
        st.warning("Please upload an image.")
    elif not instruction.strip():
        st.warning("Please enter a caption instruction.")
    else:
        # Load and show image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Generating caption with GPT-4 Turbo..."):
            # Encode image
            base64_img = encode_image(image)

            # Get caption
            caption = generate_caption(instruction.strip(), base64_img)

        # Display the generated caption
        st.subheader("📝 Generated Caption")
        st.write(caption)
