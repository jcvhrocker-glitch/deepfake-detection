from PIL import Image
import streamlit as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from src.predict import predict
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🕵️",
    layout="centered"
)

st.title("🕵️ Deepfake Detection")

st.write(
    """
    Upload a face image and the model will determine
    whether it is Real or Fake.
    """
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )

    with st.spinner("Analyzing image..."):

        result = predict(image)

    if result["label"] == "Fake":

        st.error(
            f"Prediction: {result['label']}"
        )

    else:

        st.success(
            f"Prediction: {result['label']}"
        )

    st.metric(
        "Confidence",
        f"{result['confidence']:.2%}"
    )

    st.progress(
        result["fake_probability"]
    )

    st.write(
        f"Probability of being fake: "
        f"{result['fake_probability']:.2%}"
    )
