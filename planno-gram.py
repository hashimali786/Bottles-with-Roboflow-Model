from flask import Flask, request, jsonify
from roboflow import Roboflow
import base64
import os
from datetime import datetime
from PIL import Image

app = Flask(__name__)

# # Replace with your actual Roboflow API key
# ROBOFLOW_API_KEY = ""
# PROJECT_NAME = "water-bottles"
# MODEL_VERSION = 1
#
# # Initialize Roboflow client
# rf = Roboflow(api_key=ROBOFLOW_API_KEY)
# project = rf.workspace().project(PROJECT_NAME)
# model = project.version(MODEL_VERSION).model

# rf = Roboflow(api_key="")
# project = rf.workspace().project("water_bottles")
# model = project.version(2).model

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'No selected image file'}), 400

    # if request.form.get('image') == '':
    #     return jsonify({'error': 'No image file provided'}), 400

    # For Selection of type fridge
    type_value = request.form.get('type')
    # Check if "type" is provided in the request form data
    if type_value is None:
        return jsonify({'error': 'No type provided'}), 400

    if type_value == 'singledoor':
        print("singledoor type loading...")
        # code to handle the singledoor type
        rf = Roboflow(api_key="")
        project = rf.workspace().project("water_bottles")
        model = project.version(1).model

        try:
            confidence = int(request.form.get('confidence', 40))
            overlap = int(request.form.get('overlap', 30))

            # Save the uploaded image temporarily
            image_path = "temp_image.jpg"
            image_file.save(image_path)

            # Perform prediction
            prediction_result = model.predict(image_path, confidence=confidence, overlap=overlap)
            sorted_predictions = sorted(prediction_result, key=lambda x: x['y'])

            print('sorted_predictions', sorted_predictions)

            correct_predictions = 0
            total_predictions = len(prediction_result)
            #####
            rack_product_counts = {}
            class_counts = {}
            # Iterate through the predictions
            for prediction in prediction_result:
                # Get the class name from the prediction
                class_name = prediction["class"]

                if "Rack" in class_name:
                    # Create variables to store the boundary coordinates of the "Rack" objects
                    rack_boundary_top = None
                    rack_boundary_bottom = None
                    rack_boundary_left = None
                    rack_boundary_right = None

                    # print("rack class")
                    print(f"Detected {class_name}")
                    x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
                    # print("Bounding Box Coordinates:", x, y, width, height)

                    # Update the boundary coordinates based on the current "Rack" object
                    if rack_boundary_top is None or int(y - (height / 2)) < rack_boundary_top:
                        rack_boundary_top = int(y - (height / 2))
                    if rack_boundary_bottom is None or int(y + (height)) > rack_boundary_bottom:
                        rack_boundary_bottom = int(y + (height))
                    if rack_boundary_left is None or int(x - (width / 2)) < rack_boundary_left:
                        rack_boundary_left = int(x - (width / 2))
                    if rack_boundary_right is None or int(x + (width)) > rack_boundary_right:
                        rack_boundary_right = int(x + (width))

                    # Create a dictionary to store product counts for this rack
                    product_counts = {}

                    # Loop through the predictions again to check if they are inside the boundary
                    for rk in sorted_predictions:
                        cls_n = rk['class']

                        # print("in for loop")
                        if "Rack" not in cls_n:

                            a, b, rwidth, rheight = rk['x'], rk['y'], rk['width'], rk['height']

                            # Check if the bounding box of the prediction is entirely within the boundary
                            if (
                                    a >= rack_boundary_left
                                    and b >= rack_boundary_top
                                    and (a + rwidth) <= rack_boundary_right
                                    and (b + rheight) <= rack_boundary_bottom
                            ):
                                if cls_n not in product_counts:
                                    product_counts[cls_n] = 1
                                else:
                                    product_counts[cls_n] += 1

                                # print(f"Prediction inside the boundary of {class_name}: {cls_n}")
                    # Store the product counts for this rack in the main dictionary
                    rack_product_counts[class_name] = product_counts


                # Check if the class name is already in the dictionary, if not, initialize it to 0
                if class_name not in class_counts:
                    class_counts[class_name] = 0

                # Increment the count for the class
                class_counts[class_name] += 1

            accuracy = (correct_predictions / total_predictions) * 100
            # Print the class counts
            classes_data = ''
            print()
            print('class_counts', class_counts)
            total_no_classes = len(class_counts)
            for class_name, count in class_counts.items():
                print(f"Class '{class_name}': {count} occurrences")
            print(f"Accuracy: {accuracy:.2f}%")
            print()
            expected_items = {
                "Rack 1": {"330ml_PET": 10},
                "Rack 2": {"330ml_PET": 5, "270ml_Glass_Still": 3, "270ml_Glass_Sparkling": 2},
                "Rack 3": {"330ml_PET": 5, "270ml_Glass_Still": 3, "270ml_Glass_Sparkling": 2},
                "Rack 4": {"600ml_PET": 7},
                "Rack 5": {"750ml_Glass_Still": 1, "1.5L_PET": 4, "750ml_Glass_Sparkling": 1},
            }

            # Create a dictionary to store accuracy for each rack
            product_count = {}
            rack_info = {}

            # Display the product counts for each rack
            for rack, product_count in rack_product_counts.items():
                print(f"Product Available At {rack}")
                accuracy = 0
                present_products = []
                product_accuracy = {}
                i = 0
                # Check the accuracy for each product type in this rack
                for product, count in product_count.items():
                    i = i + 1
                    # print(f"- {product}: {count}")
                    expected_count = expected_items.get(rack, {}).get(product, 0)
                    accuracy = min((count / expected_count) * 100, 100) if expected_count > 0 else 0
                    print(f"- {product}: {count} (Accuracy: {accuracy:.2f}%)")

                    # Store the accuracy for this product in this rack
                    product_accuracy[product] = accuracy
                    if count > 0:
                        # present_products.append(product)

                        present_products.append(f"{count} {product}")
                if i == 0:
                    i = i + 1
                    total_accuracy = sum(product_accuracy.values())
                    # print("total_accuracy", total_accuracy)
                    total_accuracy_percentage = (total_accuracy / i)
                    # print("i", i)
                    print(f"Total Rack Accuracy: {total_accuracy_percentage:.2f}%")
                    print()
                else:
                    total_accuracy = sum(product_accuracy.values())
                    # print("total_accuracy", total_accuracy)
                    total_accuracy_percentage = (total_accuracy / i)
                    # print("i", i)
                    print(f"Total Rack Accuracy: {total_accuracy_percentage:.2f}%")
                    print()

                rack_info[rack] = {
                    "Accuracy": total_accuracy_percentage,
                }

            overall_accuracy = sum(rack_info[rack]["Accuracy"] for rack in rack_info) / len(rack_info)
            print(f"Overall Accuracy Of Fridge: {overall_accuracy:.2f}%")

            #####
            # Generate a timestamp string using the current date and time
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            # Save the predicted image
            predicted_image_name = f'predicted_image_{timestamp}.jpg'
            # predicted_image_path = directory + str(predicted_image_name)
            print('predicted_image_path ', predicted_image_name)
            # predicted_image_path = "/var/www/html/plan-o-gram/assets/img/predicted_image.jpg"
            prediction_result.save(predicted_image_name)

            # Get the absolute path of the saved image
            # absolute_image_path = os.path.abspath(predicted_image_path)

            # Encode the predicted image as base64
            with open(predicted_image_name, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode()

            # Return the prediction result and the absolute path of the saved image
            return jsonify(
                {'image_path': predicted_image_name, 'no_of_classes': total_no_classes, 'predictions': class_counts})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    elif type_value == 'freezer':
        print("freezer type loading...")
        # code to handle the freezer type
        rf = Roboflow(api_key="")
        project = rf.workspace().project("tilt")
        model = project.version(1).model

        try:
            confidence = int(request.form.get('confidence', 40))
            overlap = int(request.form.get('overlap', 30))

            # Save the uploaded image temporarily
            image_path = "temp_image.jpg"
            image_file.save(image_path)

            # Perform prediction
            prediction_result = model.predict(image_path, confidence=confidence, overlap=overlap)

            print('prediction_result', prediction_result)

            #####
            class_counts = {}
            # Iterate through the predictions
            for prediction in prediction_result:
                # Get the class name from the prediction
                class_name = prediction["class"]

                # Check if the class name is already in the dictionary, if not, initialize it to 0
                if class_name not in class_counts:
                    class_counts[class_name] = 0

                # Increment the count for the class
                class_counts[class_name] += 1

            # Print the class counts
            classes_data = ''
            print('class_counts', class_counts)
            total_no_classes = len(class_counts)
            for class_name, count in class_counts.items():
                print(f"Class '{class_name}': {count} occurrences")
            #####
            # Generate a timestamp string using the current date and time
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            # Save the predicted image
            predicted_image_path = f'predicted_image_{timestamp}.jpg'
            # predicted_image_path = directory + str(predicted_image_name)
            print('predicted_image_path ', predicted_image_path)
            # predicted_image_path = "/var/www/html/plan-o-gram/assets/img/predicted_image.jpg"
            prediction_result.save(predicted_image_path)

            # Get the absolute path of the saved image
            absolute_image_path = os.path.abspath(predicted_image_path)

            # Encode the predicted image as base64
            with open(predicted_image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode()

            # Return the prediction result and the absolute path of the saved image
            return jsonify(
                {'image_path': absolute_image_path, 'no_of_classes': total_no_classes, 'predictions': class_counts})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        # Handle other cases or provide an error message for an unknown type
        return jsonify({'error': 'Unknown type'}), 400

    try:
        confidence = int(request.form.get('confidence', 40))
        overlap = int(request.form.get('overlap', 30))

        # Save the uploaded image temporarily
        image_path = "temp_image.jpg"
        image_file.save(image_path)

        # Perform prediction
        prediction_result = model.predict(image_path, confidence=confidence, overlap=overlap)

        print('prediction_result', prediction_result)

        #####
        class_counts = {}
        # Iterate through the predictions
        for prediction in prediction_result:
            # Get the class name from the prediction
            class_name = prediction["class"]

            # Check if the class name is already in the dictionary, if not, initialize it to 0
            if class_name not in class_counts:
                class_counts[class_name] = 0

            # Increment the count for the class
            class_counts[class_name] += 1

        # Print the class counts
        classes_data = ''
        print('class_counts', class_counts)
        total_no_classes = len(class_counts)
        for class_name, count in class_counts.items():
            print(f"Class '{class_name}': {count} occurrences")
        #####
        # Generate a timestamp string using the current date and time
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # Save the predicted image
        predicted_image_path = f'predicted_image_{timestamp}.jpg'
        # predicted_image_path = directory + str(predicted_image_name)
        print('predicted_image_path ',predicted_image_path)
        # predicted_image_path = "/var/www/html/plan-o-gram/assets/img/predicted_image.jpg"
        prediction_result.save(predicted_image_path)

        # Get the absolute path of the saved image
        absolute_image_path = os.path.abspath(predicted_image_path)

        # Encode the predicted image as base64
        with open(predicted_image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()

        # Return the prediction result and the absolute path of the saved image
        return jsonify({'image_path': absolute_image_path,'no_of_classes':total_no_classes,'predictions':class_counts})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
    # app.debug = True
    # app.run(host="0.0.0.0", port=8002)
