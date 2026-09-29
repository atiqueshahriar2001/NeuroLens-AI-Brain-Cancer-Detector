# NeuroLens AI

An educational Flask application for four-class brain MRI image classification with model-estimated uncertainty and optional visual explanation methods.

> **Research/Educational Use Only:** This application is intended for research and educational purposes only. It is not a certified medical diagnostic device and must not replace assessment by a qualified healthcare professional.

## Features

- Checkpoint-driven class order and architecture metadata, with strict PyTorch state-dict loading.
- ResNet50, EfficientNet-B0, MobileNetV3-Small, and a small CustomCNN model factory.
- CPU or CUDA inference, upload validation, and durable session-scoped SQLite prediction history.
- MC Dropout, Grad-CAM++, Captum Integrated Gradients, and EAS when available.
- Optional explanation failures do not prevent the base prediction result.

## Architecture

`app.py` provides Flask routes. `models/` builds and loads models; `inference/` validates, preprocesses, predicts, and estimates uncertainty; `explainability/` contains independent attribution helpers; templates and static assets provide the UI.

## Installation

Use Python 3.10 or newer with compatible PyTorch and Torchvision builds for your platform. Version compatibility depends on the operating system, Python, and CUDA runtime, so install the matching PyTorch/Torchvision pair from the official PyTorch instructions, then install the remaining packages:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set a private random `SECRET_KEY` for non-local use. Defaults are for local development.

Set `HISTORY_DB_PATH` to a persistent writable path when deploying. SQLite WAL mode supports concurrent readers and serialized writes on one host; use a managed database for multi-host deployments. History is capped at 100 records per browser session.

## Checkpoint setup

Put the checkpoint at `checkpoints/neurolens_best.pth`. The supplied checkpoint was inspected: metadata reports `ResNet50`, 4 outputs, and ordered classes `glioma`, `meningioma`, `notumor`, `pituitary`. Its ResNet classifier state keys support the implemented 2048→256→4 head with BatchNorm, ReLU, and Dropout. Weights are loaded with `strict=True`. If metadata or architecture does not match, loading fails with a visible error and the Flask home/about pages remain available.

Training preprocessing could not be verified from the provided files. Current default preprocessing is resize to 224×224 and ImageNet mean/std normalization. Confirm it against the training pipeline before interpreting model results.

## Running

Verify PyTorch:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Install dependencies with `pip install -r requirements.txt`, then start the app with `python app.py`. It uses Waitress as its WSGI server by default; visit http://127.0.0.1:5000. Set `FLASK_DEBUG=1` only for local development.

## Deploying to Render

This repository includes a `render.yaml` Blueprint. In Render, create a new Blueprint instance from the repository and deploy the `neurolens-ai` web service. The Blueprint installs `requirements.txt`, starts the app with Waitress, binds to `0.0.0.0`, generates a private `SECRET_KEY`, and checks `/healthz`.

The Blueprint uses Render's Free plan. Its filesystem is temporary, so prediction history in `data/history.sqlite3` can be lost when the service restarts or redeploys. Free services do not support persistent disks. To keep history, use a paid service and attach a disk mounted at `/opt/render/project/src/data`.

The supplied PyTorch model needs substantial memory. Render Free web services have 512 MB RAM, so the service may run out of memory while importing or loading the model. If that happens, use a larger memory plan. [Render's free instance limits](https://render.com/docs/free)

## Testing

Run `pytest`. Tests cover strict checkpoint loading when the checkpoint is present, output and probability shapes, CPU prediction, MC Dropout mode restoration, attribution utilities, image conversion/rejection, and Flask routes/upload validation.

## Explainability

Grad-CAM++ uses the second-order weighting formulation. Integrated Gradients uses Captum with 32 steps. The Explainability Agreement Score (EAS) is thresholded mask intersection-over-union after resizing; it compares attribution maps and is not clinical validation. Each explanation is generated independently and can be unavailable without blocking prediction.

## Uncertainty

MC Dropout enables only dropout layers while BatchNorm stays in evaluation mode, then restores evaluation. Predictive entropy and dropout probability spread describe model-estimated uncertainty, not medical certainty.

## Troubleshooting

- **Checkpoint not found:** place it at `checkpoints/neurolens_best.pth`.
- **Checkpoint mismatch:** verify architecture, classifier head, output count, and state-dict keys against checkpoint training code. Do not bypass strict loading.
- **CUDA unavailable:** CPU inference is supported; install a CUDA-matched PyTorch build to use a supported GPU.
- **CUDA out of memory:** use CPU or a smaller image; this web interface analyzes one image at a time.
- **Captum unavailable / Integrated Gradients failure:** install Captum and inspect server logs; prediction remains usable.
- **Invalid MRI / unsupported image:** use a readable JPG, JPEG, PNG, BMP, or WEBP under 16 MB.
- **Grad-CAM failure:** inspect server logs and confirm the model target layer; prediction remains usable.
- **Flask port conflict:** set `PORT` to an unused local port.
- **Missing environment variable:** `.env` is optional; set `SECRET_KEY` for deployment and install `python-dotenv` if loading `.env` through your environment.
- **Template or import error:** run from the project directory and verify all requirements are installed; see the logged traceback.

## Limitations

History is stored in SQLite and bounded to 100 items per browser session. A persistent `SECRET_KEY` is required to preserve session identity across restarts. SQLite is intended for a single host; multi-host deployments need a shared database service. The supplied checkpoint's training preprocessing, dataset, and validation procedure could not be verified from the provided files. Checkpoint `results` metadata may contain metrics, but no clinical performance claims are made here. Attribution methods can be unstable and do not explain clinical causality. There is no LLM or external API dependency.

## Medical disclaimer

This application is a research/educational prototype only. It is not a certified medical diagnostic device and cannot replace interpretation by a qualified healthcare professional.

## Future work

Verify preprocessing from the original training pipeline, add calibration and robust external validation, and assess explanation stability with documented protocols before any research conclusions.
