"""
Google Earth Engine API endpoints with pre-authentication
FastAPI routes for satellite data analysis
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import os
import ee
from google.oauth2 import service_account
import pandas as pd
import traceback
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Google Earth Engine API",
    description="Pre-authenticated API for satellite data analysis and NDVI calculations",
    version="1.0.0"
)

# Add CORS middleware for web deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global authentication state
gee_authenticated = False
auth_error = None

# Pydantic models for request/response
class NDVIAnalysisRequest(BaseModel):
    latitude: float
    longitude: float
    start_date: str
    end_date: str
    satellite: str = "sentinel-2"

class NDVIResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    count: Optional[int] = None

class ConnectionTestResponse(BaseModel):
    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None

def authenticate_gee():
    """Initialize Google Earth Engine authentication"""
    global gee_authenticated, auth_error
    
    try:
        # Get service account JSON from environment variable
        service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
        
        if not service_account_json:
            auth_error = "Service account credentials not found in environment variables"
            return False
        
        # Parse the JSON string
        try:
            service_account_info = json.loads(service_account_json)
        except json.JSONDecodeError as e:
            auth_error = f"Invalid JSON format in service account credentials: {str(e)}"
            return False
        
        # Create credentials from service account info
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=['https://www.googleapis.com/auth/earthengine']
        )
        
        # Initialize Earth Engine with credentials
        ee.Initialize(credentials)
        
        # Test the initialization
        test_val = ee.Number(42).getInfo()
        if test_val == 42:
            gee_authenticated = True
            auth_error = None
            return True
        else:
            auth_error = "Authentication test failed"
            return False
            
    except Exception as e:
        auth_error = f"Authentication failed: {str(e)}"
        return False

# Cloud masking functions
def mask_clouds_s2(image):
    """Cloud masking function for Sentinel-2"""
    qa = image.select('QA60')
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0)
    cirrus_mask = qa.bitwiseAnd(1 << 11).eq(0)
    return image.updateMask(cloud_mask.And(cirrus_mask))

def add_ndvi_s2(image):
    """Add NDVI band to Sentinel-2 image"""
    return image.addBands(
        image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    )

def mask_clouds_landsat(image):
    """Cloud masking function for Landsat"""
    qa = image.select('QA_PIXEL')
    cloud = qa.bitwiseAnd(1 << 3)
    cloud_shadow = qa.bitwiseAnd(1 << 4)
    return image.updateMask(cloud.eq(0).And(cloud_shadow.eq(0)))

def add_ndvi_landsat(image):
    """Add NDVI band to Landsat image"""
    return image.addBands(
        image.normalizedDifference(['B5', 'B4']).rename('NDVI')
    )

# Initialize authentication on startup
@app.on_event("startup")
async def startup_event():
    """Initialize GEE authentication on API startup"""
    authenticate_gee()

# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """API health check and information"""
    return {
        "message": "Google Earth Engine API",
        "status": "online",
        "authenticated": str(gee_authenticated),
        "version": "1.0.0"
    }

@app.get("/health", response_model=ConnectionTestResponse)
async def health_check():
    """Health check endpoint"""
    if not gee_authenticated:
        return ConnectionTestResponse(
            success=False,
            message=f"Not authenticated with Google Earth Engine. Error: {auth_error}"
        )
    
    try:
        # Test basic GEE connection
        test_val = ee.Number(42).getInfo()
        return ConnectionTestResponse(
            success=True,
            message="Google Earth Engine connection healthy",
            details={"test_value": test_val}
        )
    except Exception as e:
        return ConnectionTestResponse(
            success=False,
            message=f"Health check failed: {str(e)}"
        )

@app.get("/auth-status", response_model=Dict[str, Any])
async def auth_status():
    """Get authentication status"""
    return {
        "authenticated": gee_authenticated,
        "error": auth_error
    }

@app.post("/auth/refresh")
async def refresh_auth():
    """Refresh authentication"""
    success = authenticate_gee()
    return {
        "success": success,
        "authenticated": gee_authenticated,
        "error": auth_error
    }

@app.get("/test-connection", response_model=ConnectionTestResponse)
async def test_connection():
    """Test Google Earth Engine connection"""
    if not gee_authenticated:
        return ConnectionTestResponse(
            success=False,
            message="Not authenticated with Google Earth Engine"
        )
    
    try:
        # Test dataset access
        collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
        limited_collection = collection.limit(1)
        first_image = limited_collection.first()
        image_id = first_image.get('system:id').getInfo()
        
        return ConnectionTestResponse(
            success=True,
            message="Successfully connected to Google Earth Engine",
            details={"sample_image": image_id}
        )
    except Exception as e:
        return ConnectionTestResponse(
            success=False,
            message=f"Connection test failed: {str(e)}"
        )

@app.get("/datasets", response_model=Dict[str, Any])
async def test_datasets():
    """Test access to multiple datasets"""
    if not gee_authenticated:
        raise HTTPException(status_code=401, detail="Not authenticated with Google Earth Engine")
    
    datasets = {
        'Landsat 8': 'LANDSAT/LC08/C02/T1_L2',
        'Landsat 9': 'LANDSAT/LC09/C02/T1_L2', 
        'Sentinel-2 SR': 'COPERNICUS/S2_SR_HARMONIZED',
        'MODIS Terra': 'MODIS/006/MOD09GA',
        'MODIS NDVI': 'MODIS/061/MOD13Q1',
        'SRTM DEM': 'USGS/SRTMGL1_003'
    }
    
    results = {}
    
    for name, dataset_id in datasets.items():
        try:
            collection = ee.ImageCollection(dataset_id).limit(1)
            first_image = collection.first()
            image_id = first_image.get('system:id').getInfo()
            
            results[name] = {
                'success': True,
                'sample_image': image_id,
                'dataset_id': dataset_id
            }
        except Exception as e:
            results[name] = {
                'success': False,
                'error': str(e),
                'dataset_id': dataset_id
            }
    
    return {"datasets": results}

@app.post("/analyze-ndvi", response_model=NDVIResponse)
async def analyze_ndvi(request: NDVIAnalysisRequest):
    """Analyze NDVI time series for a specific location"""
    if not gee_authenticated:
        raise HTTPException(status_code=401, detail="Not authenticated with Google Earth Engine")
    
    try:
        # Create study area around the point
        point = ee.Geometry.Point([request.longitude, request.latitude])
        study_area = point.buffer(2000)  # 2km buffer
        
        if request.satellite.lower() == 'sentinel-2':
            # Sentinel-2 analysis
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterBounds(study_area)
                         .filterDate(request.start_date, request.end_date)
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                         .map(mask_clouds_s2)
                         .map(add_ndvi_s2))
        else:
            # Landsat analysis
            collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                         .filterBounds(study_area)
                         .filterDate(request.start_date, request.end_date)
                         .filter(ee.Filter.lt('CLOUD_COVER', 30))
                         .map(mask_clouds_landsat)
                         .map(add_ndvi_landsat))
        
        # Get monthly composites
        def create_monthly_composite(date):
            date = ee.Date(date)
            monthly_collection = collection.filterDate(date, date.advance(1, 'month'))
            composite = monthly_collection.median()
            return composite.set('system:time_start', date.millis())
        
        # Generate monthly dates
        start_ee = ee.Date(request.start_date)
        end_ee = ee.Date(request.end_date)
        n_months = end_ee.difference(start_ee, 'month').round()
        dates = ee.List.sequence(0, n_months).map(
            lambda n: start_ee.advance(n, 'month'))
        
        monthly_collection = ee.ImageCollection(dates.map(create_monthly_composite))
        
        # Extract time series
        time_series = monthly_collection.getRegion(point, 30).getInfo()
        
        if len(time_series) <= 1:
            return NDVIResponse(
                success=False,
                error='No data found for the specified location and time range'
            )
        
        # Process data
        df = pd.DataFrame(time_series[1:], columns=time_series[0])
        df['datetime'] = pd.to_datetime(df['time'], unit='ms')
        df['NDVI'] = pd.to_numeric(df['NDVI'], errors='coerce')
        df = df[['datetime', 'NDVI']].set_index('datetime')
        df = df.sort_index().dropna()
        
        if len(df) == 0:
            return NDVIResponse(
                success=False,
                error='No valid NDVI data found'
            )
        
        # Convert to serializable format
        data_dict = {
            'dates': df.index.strftime('%Y-%m-%d').tolist(),
            'ndvi_values': df['NDVI'].tolist(),
            'statistics': {
                'mean': float(df['NDVI'].mean()),
                'max': float(df['NDVI'].max()),
                'min': float(df['NDVI'].min()),
                'std': float(df['NDVI'].std()),
                'latest': float(df['NDVI'].iloc[-1])
            }
        }
        
        return NDVIResponse(
            success=True,
            data=data_dict,
            count=len(df)
        )
        
    except Exception as e:
        return NDVIResponse(
            success=False,
            error=f"Analysis failed: {str(e)}"
        )

@app.get("/ndvi-demo")
async def ndvi_demo():
    """Demo NDVI analysis for San Francisco"""
    demo_request = NDVIAnalysisRequest(
        latitude=37.7749,
        longitude=-122.4194,
        start_date="2023-01-01",
        end_date="2024-12-31",
        satellite="sentinel-2"
    )
    return await analyze_ndvi(demo_request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)