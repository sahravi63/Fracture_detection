"""Web app: upload an X-ray, get a fractured / not fractured prediction.  Run: python app.py"""
import gradio as gr

from classifier import FractureClassifier

clf = FractureClassifier()

DISCLAIMER = ("Research prototype. It says whether an image *looks* fractured, not where, and it is "
              "not a medical device. A radiologist must review every X-ray.")


def analyse(image, threshold):
    if image is None:
        return None, "Upload an X-ray first."
    p = clf.predict(image, threshold)
    fractured = p.label == clf.names[clf.pos]
    conf = p.p_fracture if fractured else 1 - p.p_fracture
    verdict = "FRACTURE LIKELY" if fractured else "No fracture detected"
    note = (f"**{verdict}**  \nP(fractured) = {p.p_fracture:.1%}  (threshold {threshold:.0%}, "
            f"confidence in this call {conf:.1%})")
    if abs(p.p_fracture - threshold) < 0.15:
        note += "  \nClose to the decision boundary: treat as uncertain and get a human review."
    return {n.title(): pr for n, pr in p.probs.items()}, note


with gr.Blocks(title="Bone Fracture Classifier") as demo:
    gr.Markdown("# Bone fracture X-ray classifier\n" + DISCLAIMER)
    with gr.Row():
        with gr.Column():
            img = gr.Image(type="pil", label="X-ray image", image_mode="RGB")
            thr = gr.Slider(0.05, 0.95, value=0.5, step=0.05,
                            label="Decision threshold for 'fractured' (lower = catches more fractures, more false alarms)")
            btn = gr.Button("Analyse", variant="primary")
        with gr.Column():
            label = gr.Label(label="Class probabilities", num_top_classes=2)
            text = gr.Markdown()
    btn.click(analyse, [img, thr], [label, text])
    img.change(analyse, [img, thr], [label, text])
    thr.release(analyse, [img, thr], [label, text])

if __name__ == "__main__":
    # No fixed port: Gradio starts at 7860 (or $GRADIO_SERVER_PORT) and moves to the next free one if it is busy
    demo.launch(server_name="127.0.0.1", share=True)
