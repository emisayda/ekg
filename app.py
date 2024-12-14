from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torchvision.transforms as transforms
import os


app = Flask(__name__)
CORS(app)  


MODEL_PATH = "ekg_model.pt"


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Ensure it's in the correct directory.")

device = torch.device("cpu")


model = torch.jit.load(MODEL_PATH, map_location=device)
model.eval()


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 43)),  
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        
        image = Image.open(file).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)  

        
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)

        
        classes = ["F", "M", "N", "Q", "S", "V"]  
        predicted_class = classes[predicted.item()]

        return jsonify({'class': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port)
