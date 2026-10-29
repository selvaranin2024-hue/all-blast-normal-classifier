
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

st.set_page_config(
    page_title="ALL Blast vs Normal Classifier",
    page_icon="🔬"
)

CHECKPOINT_PATH = "best_resnet18_blast_normal.pth"
DEVICE = torch.device("cpu")

@st.cache_resource
def load_model():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            "Model checkpoint is missing. Upload your .pth file."
        )

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(
            CHECKPOINT_PATH, map_location=DEVICE
        )

    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        classes = checkpoint.get(
            "class_names", ["Blast", "Normal"]
        )
        image_size = checkpoint.get("image_size", 224)
        mean = checkpoint.get(
            "mean", [0.485, 0.456, 0.406]
        )
        std = checkpoint.get(
            "std", [0.229, 0.224, 0.225]
        )
    else:
        state = checkpoint
        classes = ["Blast", "Normal"]
        image_size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    if isinstance(image_size, (tuple, list)):
        size = tuple(image_size)
    else:
        size = (image_size, image_size)

    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, len(classes))
    net.load_state_dict(state)
    net.to(DEVICE)
    net.eval()

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return net, classes, transform

st.title("🔬 ALL Blast vs Normal Classifier")
st.write("Upload a single blood cell image for classification.")

uploaded_file = st.file_uploader(
    "Choose a cell image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded cell image", width=300)

    if st.button("Classify Cell"):
        try:
            model, classes, transform = load_model()
            tensor = transform(image).unsqueeze(0).to(DEVICE)

            with torch.inference_mode():
                probabilities = torch.softmax(
                    model(tensor), dim=1
                )[0]

            index = int(probabilities.argmax())
            label = str(classes[index])
            confidence = float(probabilities[index]) * 100

            st.subheader(f"Prediction: {label}")
            st.write(f"Confidence: {confidence:.2f}%")

            st.write("Class probabilities:")
            for i, class_name in enumerate(classes):
                st.write(
                    f"{class_name}: "
                    f"{float(probabilities[i]) * 100:.2f}%"
                )

            st.warning(
                "Research prototype only; this is not a medical diagnosis."
            )

        except Exception as error:
            st.error(f"Could not load or run the model: {error}")
