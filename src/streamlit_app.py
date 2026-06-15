import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Set up the overall page configurations for Streamlit
st.set_page_config(
    page_title="Privacy-Preserving Attendance Counter",
    page_icon="📑",
    layout="wide"
)

# Project title and introductory markdown
st.title("📑 Privacy-Preserving AI Attendance Counter")
st.markdown("An automated attendance system based on head detection that counts students while protecting their privacy using smart automated blurring.")
st.markdown("---")

# Cache the model loading to prevent reloading on every user interaction
@st.cache_resource
def load_model():
    # Ensure 'best.pt' is uploaded in the same directory on Hugging Face
    return YOLO('yolov8m.pt'')

try:
    model = load_model()
except Exception as e:
    st.error("Model weights 'best.pt' not found. Please ensure it is uploaded in the same directory.")

# Split the UI layout into two equal columns: inputs and outputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 System Input")
    # File uploader widget for the user to submit classroom images
    uploaded_file = st.file_uploader("Choose or drop a classroom image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Open and display the uploaded image
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Original Uploaded Image", use_container_width=True)

with col2:
    st.subheader("🚀 Smart System Output")
    
    if uploaded_file is not None:
        # Convert the PIL image to a numpy array and then to BGR format for OpenCV compatibility
        image_np = np.array(input_image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Run inference using the YOLOv8m model with a spinner animation
        with st.spinner("Processing image and counting heads..."):
            results = model(image_bgr)
            
        # Extract detected bounding boxes and count them
        boxes = results[0].boxes
        student_count = len(boxes)
        
        # Loop through each bounding box to apply the privacy-preserving blur
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Crop the Region of Interest (ROI) corresponding to the detected head
            roi = image_bgr[y1:y2, x1:x2]
            
            if roi.size > 0:
                # Apply Gaussian Blur with a kernel size of (55, 55) to fully anonymize faces
                blurred_roi = cv2.GaussianBlur(roi, (55, 55), 0)
                image_bgr[y1:y2, x1:x2] = blurred_roi
                
                # Draw a subtle green bounding box around the blurred head to show detection
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Convert the processed BGR image back to RGB for accurate rendering in Streamlit
        output_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Display the final total count using an elegant Metric card
        st.metric(label="Total Attendance Count", value=f"{student_count} Students")
        
        # Render the final anonymized output image
        st.image(output_image, caption="Processed Image (Privacy Protected)", use_container_width=True)
        st.success("Analysis completed successfully!")
    else:
        st.info("Awaiting classroom image upload to initiate counting and privacy-preserving blurring.")
