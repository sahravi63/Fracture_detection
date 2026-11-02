"""Fracture X-ray classifier: ONNX inference shared by the CLI and the web app.

Model: YOLO11s-cls exported to ONNX, input 1x3x320x320 (RGB, 0-1), output = 2 softmax probabilities.
Class 0 = fractured, class 1 = not fractured.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True  # the dataset contains a few truncated files

DEFAULT_MODEL = Path(__file__).parent / "models" / "fracture_classifier.onnx"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


@dataclass
class Prediction:
    label: str            # "fractured" or "not fractured"
    p_fracture: float     # probability of the fractured class
    threshold: float      # decision threshold used
    probs: dict           # {class name: probability}


class FractureClassifier:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL, providers=None):
        self.session = ort.InferenceSession(
            str(model_path), providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        meta = self.session.get_modelmeta().custom_metadata_map
        self.size = ast.literal_eval(meta.get("imgsz", "[320, 320]"))[0]
        names = ast.literal_eval(meta.get("names", "{0: 'fractured', 1: 'not fractured'}"))
        self.names = [names[i] for i in sorted(names)]
        self.pos = next(i for i, n in enumerate(self.names) if "fracture" in n and not n.startswith("not"))

    def preprocess(self, img: Image.Image) -> np.ndarray:
        """Same as Ultralytics validation: EXIF fix, RGB, resize short side, centre crop, scale to 0-1."""
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        s = self.size / min(w, h)
        img = img.resize((max(self.size, round(w * s)), max(self.size, round(h * s))), Image.BILINEAR)
        w, h = img.size
        l, t = (w - self.size) // 2, (h - self.size) // 2
        img = img.crop((l, t, l + self.size, t + self.size))
        x = np.asarray(img, dtype=np.float32) / 255.0
        return x.transpose(2, 0, 1)[None]  # 1x3xHxW

    def predict(self, img: Image.Image, threshold: float = 0.5) -> Prediction:
        probs = self.session.run(None, {self.input_name: self.preprocess(img)})[0][0]
        p_fx = float(probs[self.pos])
        label = self.names[self.pos] if p_fx >= threshold else self.names[1 - self.pos]
        return Prediction(label, p_fx, threshold, {n: float(p) for n, p in zip(self.names, probs)})

    def predict_path(self, path: str | Path, threshold: float = 0.5) -> Prediction:
        with Image.open(path) as im:
            im.load()
            return self.predict(im, threshold)
