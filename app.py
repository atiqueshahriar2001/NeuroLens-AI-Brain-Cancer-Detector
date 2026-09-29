"""Flask application for NeuroLens AI research demonstrations."""
import logging
import os
import secrets
import threading
from datetime import datetime, timezone
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from config import BASE_DIR, CHECKPOINT_PATH, DEVICE, HISTORY_DB_PATH, MAX_UPLOAD_BYTES, MC_SAMPLES, SECRET_KEY
from explainability.eas import compute_eas
from explainability.gradcam import generate_gradcam_pp, overlay_heatmap
from explainability.integrated_gradients import integrated_gradients
from inference.prediction import predict_image
from inference.preprocessing import image_tensor, load_image
from models.model_loader import load_model
from utils.image_utils import image_data_url
from utils.validation import validate_filename
from utils.history_store import HistoryStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
app.config.update(SECRET_KEY=SECRET_KEY, MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES)
MAX_HISTORY = 100
history_store = HistoryStore(HISTORY_DB_PATH, MAX_HISTORY)
# Attribution code installs hooks and MC dropout changes module training flags.
# Serialize access to the shared model so concurrent requests cannot interfere.
model_lock = threading.RLock()
model = None
class_names = []
model_name = "Unavailable"
model_error = None
try:
    model, class_names, model_name = load_model(CHECKPOINT_PATH, DEVICE)
except Exception as exc:
    model_error = str(exc)
    logger.exception("Model could not be loaded; the web application remains available.")


def session_history_key():
    key = session.get("history_key")
    if not key:
        key = secrets.token_urlsafe(24)
        session["history_key"] = key
    return key


def get_history():
    return history_store.list(session_history_key())


@app.get("/")
def index():
    history = get_history()
    average_confidence = sum(row["confidence"] for row in history) / len(history) if history else 0
    class_counts = {name: sum(row["class_name"] == name for row in history) for name in class_names}
    return render_template("index.html", model_available=model is not None, model_error=model_error, model_name=model_name, class_names=class_names, mc_samples=MC_SAMPLES, records=history[:5], chart_records=list(reversed(history[:12])), class_counts=class_counts, total_scans=len(history), average_confidence=average_confidence)


@app.post("/analyze")
def analyze():
    if model is None:
        if request.accept_mimetypes.best == "application/json":
            return jsonify({"error": "The model is unavailable. Check the checkpoint and server logs."}), 503
        flash("The model is unavailable. Check that a compatible checkpoint is installed.", "error")
        return redirect(url_for("index"))
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        if request.accept_mimetypes.best == "application/json":
            return jsonify({"error": "Choose an MRI image to analyze."}), 400
        flash("Choose an MRI image to analyze.", "error")
        return redirect(url_for("index"))
    try:
        validate_filename(upload.filename)
        data = upload.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError("The image exceeds the 16 MB upload limit.")
        image = load_image(data)
        tensor = image_tensor(image, DEVICE)
        with model_lock:
            prediction = predict_image(model, tensor, class_names, DEVICE)
        result = {"prediction": prediction, "image": image_data_url(image), "explanations": {}, "uncertainty": None, "eas": None}
        try:
            with model_lock:
                uncertainty = mc_result(model, tensor)
            result["uncertainty"] = uncertainty
            # Match the notebook: the displayed prediction is argmax(mean MC
            # probabilities), with per-class dropout spread alongside it.
            mean_probs = uncertainty["mean_by_class"]
            class_index = max(range(len(class_names)), key=lambda idx: mean_probs[class_names[idx]])
            prediction = {
                "class_name": class_names[class_index],
                "class_index": class_index,
                "confidence": mean_probs[class_names[class_index]],
                "probabilities": mean_probs,
                "std_by_class": uncertainty["std_by_class"],
            }
            result["prediction"] = prediction
            uncertainty["level"], uncertainty["reliability"] = reliability_label(uncertainty["predictive_entropy"])
        except Exception as exc:
            logger.exception("MC Dropout failed.")
        cam = None
        try:
            with model_lock:
                cam = generate_gradcam_pp(model, model_name, tensor, prediction["class_index"])
            result["explanations"]["gradcam"] = image_data_url(overlay_heatmap(image, cam))
        except Exception as exc:
            logger.exception("Grad-CAM++ failed.")
        ig = None
        try:
            with model_lock:
                ig = integrated_gradients(model, tensor, prediction["class_index"])
            result["explanations"]["integrated_gradients"] = image_data_url(overlay_heatmap(image, ig))
        except Exception as exc:
            logger.exception("Integrated Gradients failed.")
        if cam is not None and ig is not None:
            try:
                result["eas"] = compute_eas(cam, ig)
            except Exception as exc:
                logger.exception("EAS calculation failed.")
        record = {"class_name": prediction["class_name"], "confidence": prediction["confidence"], "uncertainty": result["uncertainty"]["level"] if result["uncertainty"] else "Unavailable", "created": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
        history_store.add(session_history_key(), record)
        history = get_history()
        if request.accept_mimetypes.best == "application/json":
            class_counts = {name: sum(row["class_name"] == name for row in history) for name in class_names}
            return jsonify({"result": result, "record": record, "total_scans": len(history), "average_confidence": sum(row["confidence"] for row in history) / len(history), "class_counts": class_counts, "chart_records": list(reversed(history[:12]))})
        return render_template("result.html", result=result)
    except ValueError as exc:
        if request.accept_mimetypes.best == "application/json":
            return jsonify({"error": str(exc)}), 400
        flash(str(exc), "error")
        return redirect(url_for("index"))
    except Exception as exc:
        logger.exception("Image analysis failed.")
        if request.accept_mimetypes.best == "application/json":
            return jsonify({"error": "Analysis failed. Please try another image."}), 500
        flash("Analysis failed. The server log contains details.", "error")
        return redirect(url_for("index"))


def mc_result(current_model, tensor):
    from inference.uncertainty import mc_dropout
    raw = mc_dropout(current_model, tensor, MC_SAMPLES)
    raw["std_by_class"] = dict(zip(class_names, raw.pop("std_probabilities")))
    raw["mean_by_class"] = dict(zip(class_names, raw.pop("mean_probabilities")))
    return raw


def reliability_label(entropy):
    """Notebook's validation entropy cutoffs (25th and 75th percentiles)."""
    if entropy <= 0.1099:
        return "Low", "High"
    if entropy <= 0.3690:
        return "Medium", "Moderate"
    return "High", "Low"


@app.get("/history")
def history():
    return render_template("history.html", records=get_history())


@app.post("/history/clear")
def clear_history():
    history_store.clear(session_history_key())
    flash("Prediction history cleared.", "success")
    return redirect(url_for("history"))


@app.get("/about")
def about():
    return render_template("about.html", model_name=model_name, device=str(DEVICE), class_names=class_names, mc_samples=MC_SAMPLES, model_available=model is not None, model_error=model_error)


@app.errorhandler(413)
def too_large(_error):
    if request.accept_mimetypes.best == "application/json":
        return jsonify({"error": "The image exceeds the 16 MB upload limit."}), 413
    flash("The image exceeds the 16 MB upload limit.", "error")
    return redirect(url_for("index"))


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    if os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true"}:
        app.run(host=host, port=port, debug=True)
    else:
        from waitress import serve
        serve(app, host=host, port=port, threads=int(os.getenv("WAITRESS_THREADS", "4")))
