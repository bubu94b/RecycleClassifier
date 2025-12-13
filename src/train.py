from pathlib import Path  # Files management
import torch  # Neural Networks
from torch import nn, optim  # NN and optimizer
from torch.utils.data import DataLoader, random_split  # Load data by batch
from torchvision import datasets, transforms, models
from tqdm import tqdm  # Progress bar
import lzma

# Dossiers de données et de modèle
DATA_DIR = Path("data/raw/dataset-resized")  # Dossier des images TrashNet
MODEL_PATH = Path("models/recycle_resnet18.pth")  # Fichier du modèle sauvegardé

# Taille d'image attendue par ResNet18
IMG_SIZE = (224, 224)


def L_dataset():
    """
    Charge le dataset TrashNet, applique les transforms,
    split en train/val et renvoie les DataLoader + classes.
    """
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])

    # Load imgs
    dataset = datasets.ImageFolder(root=str(DATA_DIR), transform=transform)
    classes = dataset.classes

    # Splits
    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len

    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    return train_loader, val_loader, classes


def build_model(num_classes: int):
    """
    Build ResNet18 classifier with only the last layer trainable.
    """
    # ResNet18 pré-entraîné sur ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze toutes les couches sauf la dernière
    for param in model.parameters():
        param.requires_grad = False

    # Remplacer la dernière couche fully-connected
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def train_model(model, train_loader, val_loader, device, epochs: int = 3):
    """
    Train last layer and display accuracy (train & val).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

    model.to(device)

    for epoch in range(1, epochs + 1):
        # --------- TRAIN ---------
        model.train()
        total, correct = 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch} - Train"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        train_acc = correct / total
        print(f"[Epoch {epoch}] Train accuracy: {train_acc:.3f}")

        # --------- VAL ---------
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch} - Val"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()

        val_acc = correct / total
        print(f"[Epoch {epoch}] Val accuracy: {val_acc:.3f}\n")

    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device :", device)

    # Load data
    train_loader, val_loader, classes = L_dataset()
    print("Classes :", classes)

    # Build model
    model = build_model(num_classes=len(classes))

    # Train model
    model = train_model(model, train_loader, val_loader, device=device)

    # Save model
    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "classes": classes}, MODEL_PATH)
    print("Model saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()
