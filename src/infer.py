from pathlib import Path
import torch
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = Path("models/recycle_resnet18.pth")


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    classes = checkpoint["classes"]

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, len(classes))

    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, classes


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # batch=1


def predict(image_path):
    model, classes = load_model()
    x = preprocess_image(image_path)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, idx = probs.max(dim=0)

    return classes[idx.item()], float(conf.item())


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/infer.py path/to/image.jpg")
        exit(1)

    img_path = sys.argv[1]
    label, confidence = predict(img_path)
    print(f"Prediction: {label} (confidence {confidence:.2f})")

