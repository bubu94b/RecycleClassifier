# ♻️ RecycleClassifier

RecycleClassifier is a simple computer vision project that classifies waste images into recyclable categories using a convolutional neural network.

The goal of this project is to build an end-to-end machine learning pipeline:
- data loading and preprocessing
- model training using transfer learning
- inference on new images
- a lightweight web interface to test the model interactively

This project is based on the TrashNet dataset and uses PyTorch.

---

## 🧠 Model overview

The model is a **ResNet18** pre-trained on ImageNet.  
Only the final classification layer is trained on the TrashNet classes (transfer learning).

Why this choice:
- fast to train
- good performance on small datasets
- widely used baseline in computer vision projects

**Input**: RGB image resized to 224×224  
**Output**: waste category + confidence score

---

## 📁 Project structure
RecycleClassifier/
│
├── src/
│ ├── train.py # Model training script
│ └── infer.py # Inference on a single image
│
├── app/
│ └── app.py # Streamlit web application
│
├── data/ # Dataset (ignored in Git)
│
├── models/ # Trained models (ignored in Git)
│
├── requirements.txt
├── .gitignore
└── README.md

## ⚙️ Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/bubu94b/RecycleClassifier.git
cd RecycleClassifier
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Dataset 

This project uses the TrashNet dataset available on Kaggle :https://www.kaggle.com/datasets/miguem0r4/trashnet-resized-v1

 ## ⚙️ Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/bubu94b/RecycleClassifier.git
cd RecycleClassifier
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

One dataset installed, place the dataset in data/raw/dataset-resized/

Then train model on :
python src/train.py

Run a prediction with inference with a single image :
python src/infer.py path_to_image.jpg

## Streamlit app

Run the streamlit app with : streamlit run/app.py

## Author
Personal project to pratice PyTorch, CV and ML
Burak B


