from pathlib import Path
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = Path("models/recycle_resnet18.pth")
IMG_SIZE = (224, 224)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run: python src/train.py  (to create the .pth file)"
        )

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    classes = checkpoint["classes"]

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, len(classes))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, classes

def preprocess(img: Image.Image) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])
    return tfm(img.convert("RGB")).unsqueeze(0)

def predict(img: Image.Image):
    model, classes = load_model()
    x = preprocess(img)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    k = min(3, len(classes))
    top_probs, top_idxs = torch.topk(probs, k=k)
    return [(classes[i], float(p)) for p, i in zip(top_probs, top_idxs)]

def main():
    st.set_page_config(page_title="RecycleClassifier", page_icon="♻️")
    st.title("♻️ RecycleClassifier")
    st.write("Upload an image and get the predicted waste category (TrashNet).")

    uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Upload an image to start.")
        return

    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_container_width=True)

    try:
        top = predict(img)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    label, conf = top[0]
    st.subheader(f"Prediction: **{label}**")
    st.metric("Confidence", f"{conf:.2%}")

    st.write("Top predictions:")
    for l, c in top:
        st.write(f"- **{l}** — {c:.2%}")

if __name__ == "__main__":
    main()

