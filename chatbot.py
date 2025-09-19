import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import time

# Set page config
st.set_page_config(
    page_title="Image Recognition Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Cache the model loading for performance
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
        out = model.generate(**inputs, max_length=100)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def main():
    st.title("🖼️ AI Image Recognition Chatbot")
    st.markdown("Upload an image and I'll describe what I see!")
    
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("This is a hackathon project using BLIP model for image recognition. "
                "It works completely offline after the initial model download.")
        
        st.header("Instructions")
        st.write("1. Upload an image using the file uploader")
        st.write("2. Click the 'Analyze Image' button")
        st.write("3. View the results and chat history")
        
        if st.button("Clear Chat History"):
            st.session_state.history = []
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Your Image")
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            if st.button('Analyze Image', type="primary", use_container_width=True):
                with st.spinner('Analyzing image...'):
                    processor, model = load_model()
                    if processor and model:
                        start_time = time.time()
                        caption = analyze_image(image, processor, model)
                        end_time = time.time()
                        
                        # Add to history
                        st.session_state.history.append({
                            "image": image,
                            "analysis": caption,
                            "time": end_time - start_time
                        })
    
    with col2:
        st.subheader("Analysis Results")
        
        if not st.session_state.history:
            st.info("Upload an image and click 'Analyze' to see results here.")
        else:
            for i, item in enumerate(st.session_state.history):
                with st.expander(f"Analysis #{i+1} (processed in {item['time']:.2f}s)", expanded=i==len(st.session_state.history)-1):
                    st.image(item['image'], use_column_width=True)
                    st.write(item['analysis'])
                    
                    # Add a copy button for the analysis
                    if st.button("Copy Analysis", key=f"copy_{i}"):
                        st.code(item['analysis'], language=None)
                        st.success("Analysis copied to clipboard!")

if __name__ == "__main__":
    main()