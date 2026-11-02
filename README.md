# Bone Fracture X-ray Classifier

Binary image classifier (fractured / not fractured) using the trained YOLO11s-cls model exported to ONNX (`models/fracture_classifier.onnx`, input 320x320 RGB).

```
fracture_app/
  models/fracture_classifier.onnx   trained model
  classifier.py                     inference code (preprocessing + prediction)
  predict.py                        command line tool
  app.py                            web app (Gradio)
  requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Web app
```bash
python app.py        # opens on http://localhost:7860
```
Upload an X-ray to see the probabilities. The slider sets the decision threshold: lowering it (for example 0.3) catches more fractures at the cost of more false alarms, which is usually the right trade for screening.

## Command line
```bash
python predict.py xray.jpg
python predict.py folder_of_images --csv results.csv
python predict.py dataset/test --evaluate       # needs test/fractured and test/not fractured folders
python predict.py xray.jpg --threshold 0.3
```

## Run it in Google Colab
```python
!pip -q install onnxruntime gradio
# upload the fracture_app folder (or mount Drive), then:
%cd /content/fracture_app
!python app.py     # or in a cell: from app import demo; demo.launch(share=True)
```

## Notes
- Preprocessing matches training/validation: RGB, short side resized to 320, centre crop, pixels scaled to 0-1.
- The model gives an image-level answer only. It does not show where the fracture is.
- Check accuracy, sensitivity and specificity on your test set with `--evaluate` before relying on the threshold.
- Research prototype, not a validated medical device.
