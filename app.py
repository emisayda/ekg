from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torchvision.transforms as transforms
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
MODEL_PATH = "ekg_model.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Ensure it's in the correct directory.")

model = torch.jit.load(MODEL_PATH)  # Load the traced model
model.to(device)  # Move model to device
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 43)),  # Ensure the size matches your model's input
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        # Open and preprocess the image
        image = Image.open(file).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Run the model
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)

        # Map prediction to class name
        classes = ["F", "M", "N", "Q", "S", "V"]  # Replace with your actual class names
        predicted_class = classes[predicted.item()]

        return jsonify({'class': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Main entry point for the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
