# Shroomify - Mushroom Contamination Detection

🍄 **Full-stack application for mushroom contamination classification using machine learning**

## 🚀 Production-Ready Backend

**Domain:** `reliably-one-kiwi.ngrok-free.app`

### Quick Start

#### Prerequisites
- Python 3.8+
- ngrok account and authtoken
- Model files: `ann_model_state_dict.pth` and `minmax_scaler.pkl`

#### 1. Install ngrok
```bash
# Download and install ngrok from https://ngrok.com/download
# Set your authtoken
ngrok config add-authtoken YOUR_AUTHTOKEN
```

#### 2. Deploy Backend
```bash
# Windows
deploy.bat

# Linux/Mac
chmod +x deploy.sh
./deploy.sh
```

#### 3. Manual Start
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export FLASK_DEBUG=False
export PORT=5000

# Start with ngrok
python start_ngrok.py
```

## 🔧 API Endpoints

### 🏠 Home
- **URL:** `https://reliably-one-kiwi.ngrok-free.app/`
- **Method:** GET
- **Response:** API information

### 🏥 Health Check
- **URL:** `https://reliably-one-kiwi.ngrok-free.app/health`
- **Method:** GET
- **Response:** Service health status

### 📤 Image Upload
- **URL:** `https://reliably-one-kiwi.ngrok-free.app/api/upload`
- **Method:** POST
- **Content-Type:** multipart/form-data
- **Body:** `image` (file)
- **Response:**
```json
{
  "result": 0,
  "confidence": 0.95,
  "image": "base64_encoded_image",
  "status": "success"
}
```

## 🛡️ Security Features

- ✅ File type validation (PNG, JPG, JPEG, BMP, GIF)
- ✅ File size limits (8MB max)
- ✅ Rate limiting (5 requests/minute)
- ✅ Secure filename handling
- ✅ Automatic file cleanup
- ✅ Input validation
- ✅ Error handling

## 🧠 Machine Learning Features

- **YOLO Detection**: Bag detection before classification
- **GLCM Features**: Texture analysis for contamination
- **ResNet18 + CBAM**: Deep learning feature extraction
- **ANN Classification**: Final contamination prediction
- **Model Caching**: Optimized performance

## 📦 Configuration

Environment variables:
- `FLASK_DEBUG`: Enable debug mode (default: False)
- `PORT`: Server port (default: 5000)
- `HOST`: Server host (default: 0.0.0.0)
- `MAX_CONTENT_LENGTH`: Max upload size in bytes (default: 8MB)

## 🏥 Monitoring

- **Health Check:** `https://reliably-one-kiwi.ngrok-free.app/health`
- **Logs:** Check console output for detailed logs
- **Rate Limits:** 5 requests per minute per IP

## 🚀 Frontend

The repository also includes a Next.js frontend application with:
- Modern React components
- Authentication system
- Image upload interface
- Results visualization
- User profile management

## 📋 Project Structure

```
shroomify/
├── backend/                 # Flask API
│   ├── app.py              # Main application
│   ├── requirements.txt   # Python dependencies
│   ├── deploy.bat         # Windows deployment
│   ├── deploy.sh          # Linux/Mac deployment
│   └── README.md          # Backend documentation
├── src/                    # Next.js frontend
│   ├── app/               # App router pages
│   ├── lib/               # Utilities and contexts
│   └── components/         # React components
├── public/                 # Static assets
└── README.md              # This file
```

## 🔧 Development

### Backend Development
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend Development
```bash
npm install
npm run dev
```

## 🚀 Deployment

### Backend (ngrok)
```bash
cd backend
deploy.bat  # Windows
./deploy.sh # Linux/Mac
```

### Frontend (Vercel)
```bash
npm run build
vercel deploy
```

## 📊 Performance

- **Model Loading**: Cached at startup for better performance
- **File Processing**: Optimized image handling
- **Memory Management**: Automatic cleanup
- **Rate Limiting**: Prevents abuse

## 🛠️ Troubleshooting

### Model Loading Issues
```bash
# Check if model files exist
ls -la ann_model_state_dict.pth minmax_scaler.pkl best.pt

# Check logs for model loading errors
python app.py
```

### ngrok Issues
```bash
# Check ngrok status
ngrok status

# Test ngrok connection
ngrok http 5000 --domain=reliably-one-kiwi.ngrok-free.app
```

## 📈 Future Enhancements

- [ ] Docker containerization
- [ ] Database integration
- [ ] User authentication
- [ ] Batch processing
- [ ] Model versioning
- [ ] Performance monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues or questions:
1. Check the health endpoint first
2. Review application logs
3. Verify model files are present
4. Test with small image files first

---

**Ready for production deployment! 🚀**