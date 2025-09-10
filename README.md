# 🌍 Google Earth Engine Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

A comprehensive web application for satellite data analysis and NDVI (Normalized Difference Vegetation Index) monitoring using Google Earth Engine. **Ready for one-click deployment to Streamlit Cloud!**

## Features

- 🌍 **Interactive Web Dashboard**: Streamlit-based interface for data exploration
- 🔗 **REST API**: FastAPI endpoints for programmatic access
- 🛰️ **Multi-Satellite Support**: Sentinel-2 and Landsat-8 imagery
- 📊 **NDVI Analysis**: Time-series vegetation monitoring with interactive charts
- 🗺️ **Regional Analysis**: Administrative boundary-based vegetation analysis
- 🔐 **Pre-authenticated**: Secure service account authentication
- 📈 **Interactive Visualizations**: Plotly-based forex-style charts

## 🚀 One-Click Deployment

### Deploy to Streamlit Cloud (Recommended)
1. **Fork this repository** to your GitHub account
2. **Rename** `streamlit_requirements.txt` to `requirements.txt` 
3. **Deploy** at [share.streamlit.io](https://share.streamlit.io)
4. **Add secret**: `GOOGLE_SERVICE_ACCOUNT_JSON` with your service account JSON
5. **Done!** Your app will be live in minutes

### Google Earth Engine Setup (Required)
1. Create a Google Cloud Project
2. Enable the Earth Engine API  
3. Create a service account with Earth Engine permissions
4. Download the service account JSON key

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit only
streamlit run app.py --server.port 5000

# Run API only
uvicorn api:app --host 0.0.0.0 --port 8000

# Run both (combined)
python run_combined.py
```

### Deployment Options

#### 1. Streamlit Cloud

1. Push to GitHub
2. Connect to Streamlit Cloud
3. Add `GOOGLE_SERVICE_ACCOUNT_JSON` to secrets
4. Deploy from `app.py`

#### 2. Heroku

```bash
# Using Procfile (Streamlit only)
git push heroku main

# For combined deployment, update Procfile to:
# web: python run_combined.py
```

#### 3. Railway/Render

Similar to Heroku - uses `Procfile` and environment variables.

#### 4. Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000 8000

CMD ["python", "run_combined.py"]
```

## API Documentation

When running locally, access the interactive API documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

- `GET /health` - Health check and authentication status
- `POST /analyze-ndvi` - NDVI time series analysis
- `GET /test-connection` - Test Google Earth Engine connectivity
- `GET /datasets` - Test multiple satellite dataset access
- `GET /ndvi-demo` - Demo NDVI analysis for San Francisco

### Example API Usage

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# NDVI analysis
ndvi_request = {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "start_date": "2023-01-01",
    "end_date": "2024-12-31",
    "satellite": "sentinel-2"
}

response = requests.post("http://localhost:8000/analyze-ndvi", json=ndvi_request)
data = response.json()

if data["success"]:
    print(f"Found {data['count']} NDVI measurements")
    print(f"Mean NDVI: {data['data']['statistics']['mean']:.3f}")
```

## Project Structure

```
├── app.py                 # Streamlit web interface
├── api.py                 # FastAPI endpoints
├── run_combined.py        # Combined application runner
├── requirements.txt       # Python dependencies
├── runtime.txt           # Python version for deployment
├── Procfile              # Heroku deployment config
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Environment Variables

- `GOOGLE_SERVICE_ACCOUNT_JSON`: Complete service account JSON as string
- `PORT`: Port for web deployment (used by hosting platforms)

## Development

### Adding New Satellite Datasets

1. Add dataset ID to the datasets dictionary in `api.py`
2. Create cloud masking function if needed
3. Update collection filtering logic
4. Test with the `/datasets` endpoint

### Adding New Analysis Types

1. Create new Pydantic models for request/response
2. Add analysis function to `api.py`
3. Create corresponding route
4. Add UI components to `app.py`

## License

MIT License - feel free to use and modify for your projects.

## Support

For issues with Google Earth Engine setup, refer to the [official documentation](https://developers.google.com/earth-engine/guides/service_account).

For application issues, check the logs and ensure your service account has proper Earth Engine permissions.