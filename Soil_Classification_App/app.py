# app.py
import os, json
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing import image
import joblib

# -----------------------
# Paths / Config
# -----------------------
ALLOWED_EXT = {"png", "jpg", "jpeg"}
IMG_H, IMG_W = 224, 224

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(UPLOAD_DIR, exist_ok=True)

CNN_PATH   = os.path.join(MODEL_DIR, "cnn_model.keras")
PIPE_PATH  = os.path.join(MODEL_DIR, "pca_svm_pipeline.pkl")
LABELS_JS  = os.path.join(MODEL_DIR, "labels.json")
THRESH_JS  = os.path.join(MODEL_DIR, "thresholds.json")

if not (os.path.exists(CNN_PATH) and os.path.exists(PIPE_PATH)
        and os.path.exists(LABELS_JS) and os.path.exists(THRESH_JS)):
    raise RuntimeError("Missing one or more artifacts in 'models/' "
                       "(cnn_model.keras, pca_svm_pipeline.pkl, labels.json, thresholds.json)")

# -----------------------
# Load artifacts
# -----------------------
cnn  = load_model(CNN_PATH)
pipe = joblib.load(PIPE_PATH)

with open(LABELS_JS, "r") as f:
    labels = json.load(f)["labels"]

with open(THRESH_JS, "r") as f:
    TH = json.load(f)
MAX_PROB_TAU = float(TH["max_prob_tau"])
MARGIN_TAU   = float(TH["margin_tau"])
ENTROPY_TAU  = float(TH["entropy_tau"])

# Build Flatten feature extractor once
_ = cnn(np.zeros((1, IMG_H, IMG_W, 3), dtype=np.float32))  # build graph
flatten_layer = None
for lyr in cnn.layers[::-1]:
    if isinstance(lyr, Flatten):
        flatten_layer = lyr
        break
if flatten_layer is None:
    raise RuntimeError("Flatten layer not found.")
feature_extractor = Model(inputs=cnn.inputs, outputs=flatten_layer.output)

# -----------------------
# Helpers
# -----------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[-1].lower() in ALLOWED_EXT

def preprocess(img_path: str) -> np.ndarray:
    img = image.load_img(img_path, target_size=(IMG_H, IMG_W))
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def ood_decision(probs: np.ndarray):
    """
    Unknown rule: low max_prob OR low margin OR high entropy -> Unknown
    """
    top2 = np.sort(probs)[-2:]
    max_p = float(top2[-1])
    margin = float(top2[-1] - top2[-2])
    entropy = float((-probs * np.log(probs + 1e-12)).sum())
    is_unknown = (max_p < MAX_PROB_TAU) or (margin < MARGIN_TAU) or (entropy > ENTROPY_TAU)
    return is_unknown, max_p, margin, entropy

def predict_img(img_path: str):
    arr = preprocess(img_path)
    feat = feature_extractor.predict(arr, verbose=0)      # (1, 25088)
    probs = pipe.predict_proba(feat)[0]                   # (4,)

    is_unk, max_p, margin, entropy = ood_decision(probs)

    if is_unk:
        zero_probs = {cls: 0.0 for cls in labels}
        return {
            "label": "Unknown",
            "confidence": max_p,
            "probs": zero_probs,
            "max_prob": max_p,
            "margin": margin,
            "entropy": entropy
        }

    pred_idx = int(np.argmax(probs))
    return {
        "label": labels[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probs": {cls: float(p) for cls, p in zip(labels, probs)},
        "max_prob": max_p,
        "margin": margin,
        "entropy": entropy
    }

# -----------------------
# Flask
# -----------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

# serve uploaded files
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route("/", methods=["GET"])
def index():
    # initial page, no result, no filename
    return render_template("index.html", result=None, filename=None, labels=labels)

@app.route("/predict", methods=["POST"])
def predict_route():
    last_file = request.form.get("last_file") or ""   # persisted filename

    f = request.files.get("file")
    filename_for_view = None
    path = None

    if f and allowed_file(f.filename):
        fname = secure_filename(f.filename)
        path = os.path.join(UPLOAD_DIR, fname)
        f.save(path)
        filename_for_view = fname
    else:
        # no new file → reuse previous one if it exists
        if last_file and allowed_file(last_file):
            path = os.path.join(UPLOAD_DIR, last_file)
            if os.path.exists(path):
                filename_for_view = last_file

    if not path:
        # nothing valid; keep UI but no filename
        return render_template(
            "index.html",
            result={"label": "Unknown", "confidence": 0.0,
                    "probs": {l: 0.0 for l in labels},
                    "max_prob": 0.0, "margin": 0.0, "entropy": 0.0},
            filename=None, labels=labels
        )

    result = predict_img(path)
    return render_template("index.html", result=result, filename=filename_for_view, labels=labels)



# Optional JSON API
@app.route("/api/predict", methods=["POST"])
def api_predict():
    f = request.files.get("file")
    if not f or not allowed_file(f.filename):
        return jsonify({"error":"missing/invalid file"}), 400
    fname = secure_filename(f.filename)
    path = os.path.join(UPLOAD_DIR, fname)
    f.save(path)
    return jsonify(predict_img(path))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
