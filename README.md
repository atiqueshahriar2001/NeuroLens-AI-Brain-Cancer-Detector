# NeuroLens AI — Brain Tumor MRI Classification

A deep learning-powered brain tumor detection and classification system. The project trains and evaluates CNN models (Custom CNN, ResNet-50, EfficientNet-B0) on brain MRI scans to classify tumors into four categories, and ships a Streamlit web application for interactive inference with Grad-CAM explainability.

## Features

- **Multi-model classification** — compares Custom CNN, ResNet-50, and EfficientNet-B0 architectures trained on MRI brain scans
- **Four tumor classes** — Glioma, Meningioma, No Tumor, Pituitary
- **Web application** (`web_app/app.py`) built with Streamlit featuring:
  - Dark blue-black dashboard theme
  - MRI image upload & real-time prediction
  - Class-wise probability distribution & confidence scores
  - Grad-CAM explainability visualization highlighting regions of interest
  - Prediction session history with dashboard analytics
  - Context-aware AI assistant for common MRI/Grad-CAM questions
  - Settings page with system info and session reset
- **Pre-trained model** — `brain_tumor_detector.pth` included for out-of-the-box inference
- **Training notebook** — `Notebooks/Brain_Tumor_Detection_PyTorch.ipynb` with full training, evaluation, and visualization pipeline
- **Results** — confusion matrices, ROC curves, training curves, and model comparison plots in `Results/`

## Project Structure

```
Brain Cancer Detector/
├── Brain Tumor MRI Dataset/        # MRI scan dataset (JPEG images)
│   ├── Testing/                    #   Test set images
│   ├── Training/                   #   Training set images
│   └── glioma/ meningioma/ notumor/ pituitary/   # Class-organized images
├── Notebooks/
│   └── Brain_Tumor_Detection_PyTorch.ipynb   # Training & evaluation notebook
├── Results/                        # Visualization outputs
│   ├── augmentation_examples.png
│   ├── class_distribution.png
│   ├── training_{custom_cnn,efficientnet-b0,resnet50}.png
│   ├── cm_{custom_cnn,efficientnet-b0,resnet50}.png
│   ├── roc_{custom_cnn,efficientnet-b0,resnet50}.png
│   ├── model_comparison.png
│   └── sample_images.png
├── web_app/
│   ├── app.py                      # Streamlit application (NeuroLens AI)
│   └── brain_tumor_detector.pth    # Pre-trained PyTorch model checkpoint
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.8+
- See `requirements.txt` for Python dependencies

## Installation

```bash
pip install -r requirements.txt
```

### Web App Dependencies

The web application additionally relies on the following installed in `requirements.txt`:

| Package         | Purpose                      |
|-----------------|------------------------------|
| streamlit       | Web application framework    |
| torch           | Deep learning inference      |
| torchvision     | Image transforms & models    |
| pillow          | Image loading & preprocessing |
| numpy           | Numerical computations       |
| matplotlib      | Grad-CAM visualization       |

## Usage

### Run the Web Application

```bash
cd web_app
streamlit run app.py
```

Upload a brain MRI image (JPG, JPEG, PNG, or WEBP) to receive an AI-based tumor classification with probability distribution and a Grad-CAM heatmap.

### Train a Model

Open `Notebooks/Brain_Tumor_Detection_PyTorch.ipynb` and run all cells to:
1. Load and preprocess the MRI dataset
2. Apply data augmentation
3. Train Custom CNN, ResNet-50, and EfficientNet-B0
4. Evaluate with confusion matrices and ROC curves
5. Save the best model checkpoint

### Model Checkpoint Format

`brain_tumor_detector.pth` contains:
- `model_state_dict` — model weights
- `class_names` — class label list (Glioma, Meningioma, No Tumor, Pituitary)
- `num_classes` — number of output classes
- `best_model_name` — architecture name (CustomCNN, ResNet50, or EfficientNet-B0)

## Configuration

Key model parameters in `app.py`:

| Parameter    | Value                          |
|--------------|--------------------------------|
| `IMG_SIZE`   | 224 (resize target)            |
| `NORM_MEAN`  | `[0.485, 0.456, 0.406]`       |
| `NORM_STD`   | `[0.229, 0.224, 0.225]`      |
| Device        | CUDA (GPU) if available, else CPU |

## Disclaimer

NeuroLens AI is an educational and research-oriented prototype. It is **not** a substitute for professional medical diagnosis. Always consult a qualified radiologist or physician for clinical decisions.
