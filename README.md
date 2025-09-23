# Shroomify Backend - ngrok Deployment

🍄 **Mushroom Contamination Classification API**

**Domain:** `reliably-one-kiwi.ngrok-free.app`

## Quick Start

### Prerequisites
- Python 3.8+
- ngrok account and authtoken
- Model files: `ann_model_state_dict.pth` and `minmax_scaler.pkl`

### 1. Install ngrok
```bash
# Download and install ngrok from https://ngrok.com/download
# Set your authtoken
ngrok config add-authtoken YOUR_AUTHTOKEN
```

### 2. Deploy
```bash
# Make script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

### 3. Manual Start
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export FLASK_DEBUG=False
export PORT=5000

# Start with ngrok
python start_ngrok.py
```

## API Endpoints

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
  "status": "success"
}
```

## Security Features

- ✅ File type validation (PNG, JPG, JPEG, BMP, GIF)
- ✅ File size limits (8MB max)
- ✅ Rate limiting (5 requests/minute)
- ✅ Secure filename handling
- ✅ Automatic file cleanup
- ✅ Input validation
- ✅ Error handling

## Configuration

Environment variables:
- `FLASK_DEBUG`: Enable debug mode (default: False)
- `PORT`: Server port (default: 5000)
- `HOST`: Server host (default: 0.0.0.0)
- `MAX_CONTENT_LENGTH`: Max upload size in bytes (default: 8MB)

## Monitoring

- **Health Check:** `https://reliably-one-kiwi.ngrok-free.app/health`
- **Logs:** Check console output for detailed logs
- **Rate Limits:** 5 requests per minute per IP

## Troubleshooting

### Model Loading Issues
```bash
# Check if model files exist
ls -la ann_model_state_dict.pth minmax_scaler.pkl

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

### Performance Issues
- Monitor memory usage during classification
- Check rate limiting logs
- Verify model files are not corrupted

## Production Notes

- Models are loaded once at startup for better performance
- Files are automatically cleaned up after processing
- Rate limiting prevents abuse
- Comprehensive error handling and logging
- Security headers and validation

## Support

For issues or questions:
1. Check the health endpoint first
2. Review application logs
3. Verify model files are present
4. Test with small image files first
