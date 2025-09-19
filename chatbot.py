import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import time

# Set page config for mobile
st.set_page_config(
    page_title="Image Recognition Chatbot",
    page_icon="🤖",
    layout="centered",  # Better for mobile than "wide"
    initial_sidebar_state="collapsed"  # Auto-hide sidebar on mobile
)

# Cache the model loading
@st.cache_resource
def load_model():
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def analyze_image(image, processor, model):
    try:
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_length=100, num_beams=5, early_stopping=True)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def main():
    st.title("🖼️ AI Image Chatbot")
    
    # Mobile-friendly layout
    st.subheader("Upload or Capture Image")
    
    # Mobile-friendly file input options
    option = st.radio("Choose input method:", 
                     ["Upload from device", "Take photo with camera"],
                     horizontal=True)  # Horizontal looks better on mobile
    
    uploaded_file = None
    if option == "Take photo with camera":
        uploaded_file = st.camera_input("Take a picture", label_visibility="collapsed")
    else:
        uploaded_file = st.file_uploader("Choose image file", 
                                       type=["jpg", "jpeg", "png"],
                                       label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Your Image', use_column_width=True)
        
        if st.button('Analyze Image', type="primary", use_container_width=True):
            with st.spinner('Analyzing...'):
                processor, model = load_model()
                if processor and model:
                    start_time = time.time()
                    caption = analyze_image(image, processor, model)
                    end_time = time.time()
                    
                    # Display results
                    st.success("Analysis Complete!")
                    st.subheader("What I see:")
                    st.write(caption)
                    st.caption(f"Processed in {end_time - start_time:.2f} seconds")
                    
                    # Copy to clipboard option
                    if st.button("Copy Analysis", use_container_width=True):
                        st.code(caption, language=None)
                        st.success("Copied to clipboard!")

# Hide sidebar on mobile for more space
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
/* Mobile-specific styles */
@media (max-width: 768px) {
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton button {
        width: 100%;
    }
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
