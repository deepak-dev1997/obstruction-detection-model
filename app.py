import os
from io import BytesIO
from flask import Flask, request, render_template_string, send_file
import cv2
import numpy as np
from ultralytics import YOLO

# Load the trained YOLO model (ensure obstruction_yolo.pt exists)
model_path = "./obstruction_yolo.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model weights not found at {model_path}. Train the model first.")
model = YOLO(model_path)

# Define a basic HTML template for file upload.
HTML_TEMPLATE = """
<!doctype html>
<title>Rooftop Obstruction Detection</title>
<h1>Upload an image for obstruction detection</h1>
<form method=post enctype=multipart/form-data action="/predict">
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
"""

app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file part.", 400
    file = request.files["file"]
    if file.filename == "":
        return "No selected file.", 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Predict (use a moderate confidence)
    results = model.predict(img, conf=0.3, verbose=False)

    annotated_img = img.copy()
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()

    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_img, f'{conf:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    _, buffer = cv2.imencode('.png', annotated_img)
    io_buf = BytesIO(buffer)

    return send_file(io_buf, mimetype='image/png')

if __name__ == "__main__":
    app.run(port=5010,debug=False)
