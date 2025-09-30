from flask import Flask, render_template, request, send_from_directory
import os
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# Load the YOLO model
model = YOLO('lovey.pt')

# Route to display the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    # Save the uploaded file temporarily
    img = Image.open(file.stream)

    # Run the YOLO model for inference
    results = model.predict(source=img, conf=0.5)

    # Get the first result (for simplicity)
    for r in results:
        im_array = r.plot()  # Get the image with bounding boxes
        im = Image.fromarray(im_array[..., ::-1])  # Convert to PIL image
        img_path = os.path.join('static', 'output.jpg')
        im.save(img_path)  # Save the result image in static folder

    return render_template('result.html', img_path='output.jpg')

# Serve the images from the static folder
@app.route('/templates/<filename>')
def serve_image(filename):
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)

if __name__ == '__main__':
    app.run(debug=True)
