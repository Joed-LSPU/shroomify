import joblib
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from skimage.color import rgb2gray
from torchvision.models import resnet18
from skimage.feature import graycomatrix, graycoprops
from torchvision.models import resnet18, ResNet18_Weights
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64

app = Flask(__name__)
CORS(app)

glcm_features = []
resnet_features = []
pca_features = []
image_name = []
classification = []


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, 'minmax_scaler.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'ann_model_state_dict.pth')

import os
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops

class GLCMFeatureExtractor:
    def __init__(self, distances=[50], angles=[np.pi/2], levels=256, props=None):
        self.distances = distances
        self.angles = angles
        self.levels = levels
        self.props = props or ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        self.feature_names = [f"{prop.capitalize()}" for prop in self.props]

    def extract_from_image(self, image_path):
        image = self._load_image(image_path)
        glcm = graycomatrix(image, 
                            distances=self.distances,
                            angles=self.angles,
                            levels=self.levels)
        
        features = {}
        for prop in self.props:
            value = graycoprops(glcm, prop)[0, 0]
            features[prop.capitalize()] = value
        
        return features

    def _load_image(self, image_path):
        im_frame = Image.open(image_path).convert("RGB")
        gray_image = rgb2gray(np.array(im_frame))
        return (gray_image * 255).astype(np.uint8)

# Use the extractor
def classify_contamination(img_path):
    extractor = GLCMFeatureExtractor()
    return extractor.extract_from_image(img_path)

import torch, random, numpy as np

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // ratio, in_planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).flatten(1))
        max_out = self.fc(self.max_pool(x).flatten(1))
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x
    
class ResNet18_CBAM_FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            CBAM(64),
            base_model.layer2,
            CBAM(128),
            base_model.layer3,
            CBAM(256),
            base_model.layer4,
            CBAM(512)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)
    
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18_CBAM_FeatureExtractor().to(device)
model.eval()

def extract_deep_features(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img_tensor).cpu().squeeze(0).numpy()
    return features

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(517, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(128, 3)  # adjust to 2 for binary, more for multiclass

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return self.output(x)
    
def classify(img_path):
    img_path = img_path
    glcm_features = list(classify_contamination(img_path).values())
    resnet_features = extract_deep_features(img_path)
    
    scaler = joblib.load(SCALER_PATH)
    features = np.concatenate((glcm_features, resnet_features), axis=0).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    ann_model = ANN()
    ann_model.load_state_dict(torch.load(MODEL_PATH))
    ann_model.eval()

    input_tensor = torch.tensor(features_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = ann_model(input_tensor)
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()

    return predicted_class, confidence

@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    image = request.files['image']

    # Ensure upload folder exists
    os.makedirs('uploads', exist_ok=True)
    image_path = os.path.join('uploads', image.filename)
    image.save(image_path)

    img = Image.open(image_path)
    img = img.resize((224, 224))

    img.save(image_path)

    _class, _confidence = classify(image_path)

    with open(image_path, "rb") as f:
        img_bytes = f.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    return jsonify({'result': _class,
                    'confidence': _confidence,
                    'image': img_base64})

if __name__ == '__main__':
    app.run(debug=True)
