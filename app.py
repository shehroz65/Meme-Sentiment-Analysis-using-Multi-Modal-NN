from flask import Flask, render_template, request, redirect, url_for
import torch
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)

# Load the saved model
combined_model = torch.load("combined_model.pth")
combined_model.eval()  # Set the model to evaluation mode

# Transformation pipeline for preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        image = Image.open(file.stream).convert('RGB')
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            sentiment_output, humor_output = combined_model(image)
            _, sentiment_predicted = torch.max(sentiment_output, 1)
            _, humor_predicted = torch.max(humor_output, 1)
            
        sentiments = ["Positive", "Negative"]  # Replace with your classes if different
        humors = ["Funny", "Not Funny"]        # Replace with your classes if different
        sentiment = sentiments[sentiment_predicted[0]]
        humor = humors[humor_predicted[0]]
    
        return render_template('result.html', sentiment=sentiment, humor=humor)

if __name__ == '__main__':
    app.run(debug=True)
