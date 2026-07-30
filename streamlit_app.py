import os
import tempfile

import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# Page configuration
st.set_page_config(
    page_title="Privacy-Preserving Attendance Counter",
    page_icon="📑",
    layout="wide",
)


# Title and description
st.title("📑 Privacy-Preserving AI Attendance Counter")
st.markdown(
    "A smart attendance system that counts attendance automatically "
    "while keeping everyone’s privacy protected."
)
st.markdown("---")


# Load and cache the trained model
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")


try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()


# Page layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 System Input")

    uploaded_file = st.file_uploader(
        "Choose or drop an image...",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension,
        ) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        input_image = Image.open(temp_path).convert("RGB")

        st.image(
            input_image,
            caption="Original Uploaded Image",
            use_container_width=True,
        )


with col2:
    st.subheader("🚀 Smart System Output")

    if uploaded_file is not None:
        image_bgr = cv2.imread(temp_path)

        if image_bgr is None:
            st.error("The uploaded image could not be read.")
            st.stop()

        with st.spinner("Analyzing attendance and protecting privacy..."):
            results = model.predict(
                source=temp_path,
                conf=0.25,
                iou=0.30,
                classes=[0],
                imgsz=640,
                verbose=False,
            )

        boxes = results[0].boxes
        attendance_count = len(boxes)

        for box in boxes:
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].cpu().tolist(),
            )

            height, width = image_bgr.shape[:2]

            x1 = max(0, min(x1, width))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))

            roi = image_bgr[y1:y2, x1:x2]

            if roi.size > 0:
                blurred_roi = cv2.GaussianBlur(
                    roi,
                    (51, 51),
                    30,
                )

                image_bgr[y1:y2, x1:x2] = blurred_roi

                cv2.rectangle(
                    image_bgr,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

        output_image = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        st.metric(
            label="Total Attendance Count",
            value=attendance_count,
        )

        st.image(
            output_image,
            caption="Processed Image (Privacy Protected)",
            use_container_width=True,
        )

        st.success("Analysis completed successfully!")

        try:
            os.remove(temp_path)
        except OSError:
            pass

    else:
        st.info(
            "Upload an image to begin attendance counting "
            "and privacy-preserving processing."
        )