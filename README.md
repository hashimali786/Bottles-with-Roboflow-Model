# Bottles-with-Roboflow-Model
How the Code Works:
# 1. Setup and Imports
python
Copy
Edit
from flask import Flask, request, jsonify
from roboflow import Roboflow
import base64
import os
from datetime import datetime
from PIL import Image
Flask: Used to create a web server for handling image uploads and processing requests.
Roboflow: A cloud-based object detection API used for predicting objects in images.
Base64: Used to encode the processed image in base64 format.
OS: Handles file operations like saving images temporarily.
Datetime: Generates timestamps for file naming.
PIL (Pillow): Image processing library (though not actively used in the provided code).
# 2. Flask App Initialization
python
Copy
Edit
app = Flask(__name__)
Initializes a Flask application to create API routes.
3. Prediction API Route
python
Copy
Edit
@app.route('/predict', methods=['POST'])
def predict():
Defines a POST API endpoint /predict, which processes images sent in the request.
# 4. Image Validation
python
Copy
Edit
if 'image' not in request.files:
    return jsonify({'error': 'No image file provided'}), 400
Ensures an image file is provided in the request.
If not, returns an error response.
python
Copy
Edit
image_file = request.files['image']

if image_file.filename == '':
    return jsonify({'error': 'No selected image file'}), 400
Ensures that a file name is provided.
# 5. Fridge Type Selection
python
Copy
Edit
type_value = request.form.get('type')
if type_value is None:
    return jsonify({'error': 'No type provided'}), 400
Checks if the "type" parameter is provided (either "singledoor" or "freezer").
If missing, returns an error.
Handling ‘singledoor’ Type
# 6. Load Roboflow Model
python
Copy
Edit
if type_value == 'singledoor':
    print("singledoor type loading...")
    rf = Roboflow(api_key="")
    project = rf.workspace().project("water_bottles")
    model = project.version(1).model
Loads a Roboflow model trained to detect water bottles in a single-door fridge.
# 7. Retrieve Confidence and Overlap Parameters
python
Copy
Edit
confidence = int(request.form.get('confidence', 40))
overlap = int(request.form.get('overlap', 30))
Confidence threshold (default 40%): Minimum confidence required for detections.
Overlap threshold (default 30%): Determines how much objects can overlap before being merged.
# 8. Save the Image Temporarily
python
Copy
Edit
image_path = "temp_image.jpg"
image_file.save(image_path)
Saves the uploaded image to disk as "temp_image.jpg" for processing.
# 9. Run Object Detection
python
Copy
Edit
prediction_result = model.predict(image_path, confidence=confidence, overlap=overlap)
sorted_predictions = sorted(prediction_result, key=lambda x: x['y'])
Runs object detection on the image using Roboflow.
Sorts detections based on Y-coordinates to analyze rack-level placement.
Processing Predictions
# 10. Counting Detected Objects
python
Copy
Edit
class_counts = {}
rack_product_counts = {}
for prediction in prediction_result:
    class_name = prediction["class"]

    if class_name not in class_counts:
        class_counts[class_name] = 0
    class_counts[class_name] += 1
Counts occurrences of each detected object class.
# 11. Rack-wise Product Classification
python
Copy
Edit
if "Rack" in class_name:
    rack_boundary_top = int(y - (height / 2))
    rack_boundary_bottom = int(y + (height))
    rack_boundary_left = int(x - (width / 2))
    rack_boundary_right = int(x + (width))

    product_counts = {}
    for rk in sorted_predictions:
        cls_n = rk['class']
        if "Rack" not in cls_n:
            a, b, rwidth, rheight = rk['x'], rk['y'], rk['width'], rk['height']
            if (
                a >= rack_boundary_left and 
                b >= rack_boundary_top and 
                (a + rwidth) <= rack_boundary_right and 
                (b + rheight) <= rack_boundary_bottom
            ):
                if cls_n not in product_counts:
                    product_counts[cls_n] = 1
                else:
                    product_counts[cls_n] += 1

    rack_product_counts[class_name] = product_counts
Identifies racks in the image and counts bottles within each rack.
# 12. Expected Items vs Detected Items
python
Copy
Edit
expected_items = {
    "Rack 1": {"330ml_PET": 10},
    "Rack 2": {"330ml_PET": 5, "270ml_Glass_Still": 3, "270ml_Glass_Sparkling": 2},
}
Defines the expected count of bottles per rack.
python
Copy
Edit
for rack, product_count in rack_product_counts.items():
    total_accuracy = sum(min((count / expected_items.get(rack, {}).get(product, 0)) * 100, 100)
                         for product, count in product_count.items())
    rack_info[rack] = {"Accuracy": total_accuracy}
Compares detected products vs expected products per rack and calculates accuracy.
# 13. Image Processing and Base64 Encoding
python
Copy
Edit
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
predicted_image_name = f'predicted_image_{timestamp}.jpg'
prediction_result.save(predicted_image_name)
Saves the annotated image with detected objects.
python
Copy
Edit
with open(predicted_image_name, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()
Converts the predicted image to Base64 format for API response.
# 14. Return the JSON Response
python
Copy
Edit
return jsonify({
    'image_path': predicted_image_name, 
    'no_of_classes': len(class_counts), 
    'predictions': class_counts
})
Sends JSON response containing:
Image path
Number of detected classes
Count of each detected class
Handling ‘freezer’ Type
python
Copy
Edit
elif type_value == 'freezer':
    print("freezer type loading...")
    rf = Roboflow(api_key="")
    project = rf.workspace().project("tilt")
    model = project.version(1).model
Uses a different Roboflow model trained to detect bottles in a freezer.
Error Handling
python
Copy
Edit
except Exception as e:
    return jsonify({'error': str(e)}), 500
Catches and returns any unexpected errors.
Summary
Receives an image via POST request.
Checks if the image is valid and retrieves parameters (type, confidence, overlap).
Loads a specific Roboflow model based on the fridge type (singledoor or freezer).
Runs object detection on the image.
Sorts objects and groups them by racks.
Compares detected vs expected items, calculating accuracy.
Saves the annotated image and encodes it in Base64.
Returns JSON response with predictions.
