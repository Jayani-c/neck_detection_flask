import os
import torch
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('yolov5', 'custom', path='bestn.pt', source='local')
model.eval()

@app.route('/detect', methods=['POST'])
def detect_objects():
    try:
        # Get uploaded image from the request
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        image = request.files['image']
        image_path = os.path.join('uploads', image.filename)
        image.save(image_path)
 

        # Perform inference
        results = model(image_path)
        predictions = results.pandas().xyxy[0]

       

        # Get object counts
        object_counts = predictions['name'].value_counts().to_dict()
        return jsonify(object_counts), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')   

    app.run(host='0.0.0.0', port=5000)



# import torch

# model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')
# # Image
# img = '2.jpg'
# # Inference
# results = model(img)
# # Results, change the flowing to: results.show()
# results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
# results.xyxy[0]  # im predictions (tensor)
# print(results.pandas().xyxy[0])  # im predictions (pandas)