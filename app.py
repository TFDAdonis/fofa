import streamlit as st
import json
import os
import ee
from google.oauth2 import service_account
import traceback
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Google Earth Engine Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'gee_authenticated' not in st.session_state:
    st.session_state.gee_authenticated = False
if 'auth_error' not in st.session_state:
    st.session_state.auth_error = None

def authenticate_gee_from_secrets():
    """
    Authenticate Google Earth Engine using service account credentials from environment variables
    
    Returns:
        bool: True if authentication successful, False otherwise
    """
    try:
        # Get service account JSON from environment variable
        service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
        
        if not service_account_json:
            st.session_state.auth_error = "🔐 Service account credentials not found. Please add GOOGLE_SERVICE_ACCOUNT_JSON to your app secrets/environment variables."
            return False
        
        # Parse the JSON string
        try:
            service_account_info = json.loads(service_account_json)
        except json.JSONDecodeError as e:
            st.session_state.auth_error = f"Invalid JSON format in service account credentials: {str(e)}"
            return False
        
        # Create credentials from service account info
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=['https://www.googleapis.com/auth/earthengine']
        )
        
        # Initialize Earth Engine with credentials
        ee.Initialize(credentials)
        
        # Test the initialization immediately
        test_val = ee.Number(42).getInfo()
        if test_val == 42:
            st.session_state.gee_authenticated = True
            st.session_state.auth_error = None
            return True
        else:
            st.session_state.auth_error = "Authentication appears successful but test failed"
            return False
            
    except Exception as e:
        st.session_state.auth_error = f"Authentication failed: {str(e)}"
        return False

def test_gee_connection():
    """
    Test Google Earth Engine connection by accessing a simple dataset
    
    Returns:
        tuple: (success_bool, message_string)
    """
    try:
        # Test 1: Check if EE is initialized
        try:
            ee.Number(1).getInfo()
        except Exception as e:
            return False, f"Earth Engine not initialized properly: {str(e)}"
        
        # Test 2: Try to access a simple dataset - Landsat 8 collection
        collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
        
        # Test 3: Get a limited size (faster)
        limited_collection = collection.limit(1)
        first_image = limited_collection.first()
        
        # Test 4: Get basic info without full collection size
        image_id = first_image.get('system:id').getInfo()
        
        return True, f"Successfully connected to Google Earth Engine! Sample image: {image_id}"
    
    except Exception as e:
        return False, f"Failed to access Google Earth Engine data: {str(e)}"

def test_data_retrieval():
    """
    Test advanced data retrieval capabilities
    
    Returns:
        dict: Results from various data retrieval tests
    """
    results = {}
    
    try:
        # Test 1: Basic collection access
        collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').limit(100)
        
        # Test 2: Get first image 
        first_image = collection.first()
        results['first_image_id'] = first_image.get('system:id').getInfo()
        
        # Test 3: Get band names
        band_names = first_image.bandNames().getInfo()
        results['band_names'] = band_names[:5]  # First 5 bands
        results['total_bands'] = len(band_names)
        
        # Test 4: Test date filtering (small sample)
        recent_images = collection.filterDate('2024-01-01', '2024-01-31').limit(5)
        results['jan_2024_sample_count'] = recent_images.size().getInfo()
        
        # Test 5: Test geographic filtering
        san_francisco = ee.Geometry.Point([-122.4194, 37.7749])
        sf_images = collection.filterBounds(san_francisco).limit(3)
        results['sf_sample_count'] = sf_images.size().getInfo()
        
        # Test 6: Get cloud cover property
        if sf_images.size().getInfo() > 0:
            sample_image = sf_images.first()
            cloud_cover = sample_image.get('CLOUD_COVER').getInfo()
            results['sample_cloud_cover'] = cloud_cover
        
        results['success'] = True
        results['message'] = "Advanced data retrieval tests completed successfully!"
        
    except Exception as e:
        results['success'] = False
        results['error'] = str(e)
        results['message'] = f"Data retrieval test failed: {str(e)}"
    
    return results

def test_multiple_datasets():
    """
    Test access to multiple different datasets
    
    Returns:
        dict: Results from testing various datasets
    """
    datasets = {
        '🛰️ Landsat 8': 'LANDSAT/LC08/C02/T1_L2',
        '🛰️ Landsat 9': 'LANDSAT/LC09/C02/T1_L2', 
        '🛰️ Sentinel-2 SR': 'COPERNICUS/S2_SR_HARMONIZED',
        '🌍 MODIS Terra': 'MODIS/006/MOD09GA',
        '🌿 MODIS NDVI': 'MODIS/061/MOD13Q1',
        '🌡️ MODIS LST': 'MODIS/061/MOD11A1',
        '📡 Sentinel-1 SAR': 'COPERNICUS/S1_GRD',
        '🏔️ SRTM DEM': 'USGS/SRTMGL1_003',
        '🌧️ CHIRPS Precipitation': 'UCSB-CHG/CHIRPS/DAILY',
        '🔥 MODIS Burned Area': 'MODIS/061/MCD64A1'
    }
    
    results = {}
    
    for name, dataset_id in datasets.items():
        try:
            # Limit collection size to avoid timeouts
            collection = ee.ImageCollection(dataset_id).limit(10)
            
            # Just check if we can access the first image
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
    
    return results

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

def analyze_ndvi_location(lat, lon, start_date, end_date, satellite):
    """Analyze NDVI time series for a specific location"""
    try:
        # Create study area around the point
        point = ee.Geometry.Point([lon, lat])
        study_area = point.buffer(2000)  # 2km buffer
        
        if satellite.lower() == 'sentinel-2':
            # Sentinel-2 analysis
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterBounds(study_area)
                         .filterDate(start_date, end_date)
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                         .map(mask_clouds_s2)
                         .map(add_ndvi_s2))
        else:
            # Landsat analysis
            collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                         .filterBounds(study_area)
                         .filterDate(start_date, end_date)
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
        start_ee = ee.Date(start_date)
        end_ee = ee.Date(end_date)
        n_months = end_ee.difference(start_ee, 'month').round()
        dates = ee.List.sequence(0, n_months).map(
            lambda n: start_ee.advance(n, 'month'))
        
        monthly_collection = ee.ImageCollection(dates.map(create_monthly_composite))
        
        # Extract time series
        time_series = monthly_collection.getRegion(point, 30).getInfo()
        
        if len(time_series) <= 1:
            return {
                'success': False,
                'error': 'No data found for the specified location and time range'
            }
        
        # Process data
        df = pd.DataFrame(time_series[1:], columns=time_series[0])
        df['datetime'] = pd.to_datetime(df['time'], unit='ms')
        df['NDVI'] = pd.to_numeric(df['NDVI'], errors='coerce')
        df = df[['datetime', 'NDVI']].set_index('datetime')
        df = df.sort_index().dropna()
        
        if len(df) == 0:
            return {
                'success': False,
                'error': 'No valid NDVI data found'
            }
        
        return {
            'success': True,
            'data': df,
            'count': len(df)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def create_interactive_ndvi_chart(df, title):
    """Create an interactive NDVI chart using Plotly"""
    fig = go.Figure()
    
    # Add NDVI line
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['NDVI'],
        mode='lines+markers',
        name='NDVI',
        line=dict(color='#00ff00', width=2),
        marker=dict(size=6, color='#00ff00'),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 0, 0.1)',
        hovertemplate='<b>Date:</b> %{x}<br><b>NDVI:</b> %{y:.3f}<extra></extra>'
    ))
    
    # Add moving average if enough data
    if len(df) > 3:
        sma = df['NDVI'].rolling(window=3, center=True).mean()
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=sma,
            mode='lines',
            name='3-month Average',
            line=dict(color='#ff9900', width=2, dash='dash'),
            hovertemplate='<b>Date:</b> %{x}<br><b>3-month Avg:</b> %{y:.3f}<extra></extra>'
        ))
    
    # Add horizontal reference lines
    mean_ndvi = df['NDVI'].mean()
    fig.add_hline(y=mean_ndvi, line_dash="dot", line_color="white", 
                  annotation_text=f"Mean: {mean_ndvi:.3f}")
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color='white'),
            x=0.5
        ),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        xaxis=dict(
            title=dict(text='Date', font=dict(color='white')),
            tickfont=dict(color='white'),
            gridcolor='#2a2a2a',
            zerolinecolor='#2a2a2a'
        ),
        yaxis=dict(
            title=dict(text='NDVI', font=dict(color='white')),
            tickfont=dict(color='white'),
            gridcolor='#2a2a2a',
            zerolinecolor='#2a2a2a',
            tickformat='.3f',
            range=[-0.1, 1.0]
        ),
        legend=dict(
            font=dict(color='white'),
            bgcolor='rgba(0,0,0,0.5)'
        ),
        hovermode='x unified',
        height=500
    )
    
    return fig

def get_admin_boundaries(level, country_code=None, admin1_code=None):
    """Get administrative boundaries at different levels"""
    FAO_GAUL = ee.FeatureCollection("FAO/GAUL/2015/level0")  # Countries
    FAO_GAUL_ADMIN1 = ee.FeatureCollection("FAO/GAUL/2015/level1")  # Admin1 (states/provinces)
    FAO_GAUL_ADMIN2 = ee.FeatureCollection("FAO/GAUL/2015/level2")  # Admin2 (municipalities)
    
    if level == 0:  # Countries
        return FAO_GAUL
    elif level == 1:  # Admin1 (states/provinces)
        if country_code:
            return FAO_GAUL_ADMIN1.filter(ee.Filter.eq('ADM0_CODE', country_code))
        return FAO_GAUL_ADMIN1
    elif level == 2:  # Admin2 (municipalities)
        if admin1_code:
            return FAO_GAUL_ADMIN2.filter(ee.Filter.eq('ADM1_CODE', admin1_code))
        return FAO_GAUL_ADMIN2
    return None

def get_boundary_names(fc, level):
    """Get names of boundaries in a feature collection for a specific level"""
    try:
        if level == 0:  # Countries
            names = fc.aggregate_array('ADM0_NAME').getInfo()
        elif level == 1:  # Admin1 (states/provinces)
            names = fc.aggregate_array('ADM1_NAME').getInfo()
        elif level == 2:  # Admin2 (municipalities)
            names = fc.aggregate_array('ADM2_NAME').getInfo()
        else:
            names = []
        return sorted(list(set(names)))  # Remove duplicates and sort
    except Exception as e:
        st.error(f"Error getting boundary names: {str(e)}")
        return []

def analyze_region_ndvi(region_fc, start_date, end_date, satellite):
    """Analyze NDVI for a region"""
    try:
        if satellite.lower() == 'sentinel-2':
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterBounds(region_fc.geometry())
                         .filterDate(start_date, end_date)
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                         .map(mask_clouds_s2)
                         .map(add_ndvi_s2))
        else:
            collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                         .filterBounds(region_fc.geometry())
                         .filterDate(start_date, end_date)
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
        start_ee = ee.Date(start_date)
        end_ee = ee.Date(end_date)
        n_months = end_ee.difference(start_ee, 'month').round()
        dates = ee.List.sequence(0, n_months).map(
            lambda n: start_ee.advance(n, 'month'))
        
        monthly_collection = ee.ImageCollection(dates.map(create_monthly_composite))
        
        # Calculate mean NDVI for the region for each month
        def calculate_region_mean(image):
            mean_ndvi = image.select('NDVI').reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region_fc.geometry(),
                scale=500,
                maxPixels=1e9
            )
            return ee.Feature(None, {
                'date': image.get('system:time_start'),
                'ndvi': mean_ndvi.get('NDVI')
            })
        
        time_series = monthly_collection.map(calculate_region_mean)
        time_series_info = time_series.getInfo()
        
        # Process data
        data = []
        for feature in time_series_info['features']:
            props = feature['properties']
            if props['ndvi'] is not None:
                date = pd.to_datetime(props['date'], unit='ms')
                data.append({'datetime': date, 'NDVI': props['ndvi']})
        
        if not data:
            return {
                'success': False,
                'error': 'No valid NDVI data found for the region'
            }
        
        df = pd.DataFrame(data).set_index('datetime').sort_index()
        
        return {
            'success': True,
            'data': df,
            'count': len(df)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# Automatic authentication on app startup
if not st.session_state.gee_authenticated:
    with st.spinner("🔐 Authenticating with Google Earth Engine..."):
        authenticate_gee_from_secrets()

# Deployment Information
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Deployment")
st.sidebar.markdown("This app is ready for:")
st.sidebar.markdown("- **Streamlit Cloud**: One-click deployment from GitHub")
st.sidebar.markdown("- **API Access**: FastAPI endpoints available when deployed with both services")
st.sidebar.markdown("- **Environment Variables**: Uses GOOGLE_SERVICE_ACCOUNT_JSON secret")

# Main app UI
st.title("🌍 Google Earth Engine Dashboard")
st.write("Satellite data analysis and visualization platform")

# Authentication status indicator
col1, col2 = st.columns([3, 1])

with col1:
    if st.session_state.gee_authenticated:
        st.success("✅ Connected to Google Earth Engine")
    else:
        st.error("❌ Not connected to Google Earth Engine")
        if st.session_state.auth_error:
            st.error(f"Error: {st.session_state.auth_error}")

with col2:
    if st.button("🔄 Retry Authentication"):
        st.session_state.gee_authenticated = False
        st.session_state.auth_error = None
        st.rerun()

# Only show the rest of the app if authenticated
if st.session_state.gee_authenticated:
    
    # Sidebar for navigation
    st.sidebar.title("🛰️ Analysis Tools")
    
    analysis_mode = st.sidebar.selectbox(
        "Select Analysis Mode",
        ["🔍 Connection Testing", "📊 NDVI Point Analysis", "🗺️ NDVI Regional Analysis", "🧪 Dataset Access Test"]
    )
    
    if analysis_mode == "🔍 Connection Testing":
        st.header("🔍 Google Earth Engine Connection Testing")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧪 Test Basic Connection"):
                with st.spinner("Testing connection..."):
                    success, message = test_gee_connection()
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        
        with col2:
            if st.button("📊 Test Data Retrieval"):
                with st.spinner("Testing data retrieval..."):
                    results = test_data_retrieval()
                    if results['success']:
                        st.success(results['message'])
                        
                        st.write("**Retrieval Results:**")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Total Bands", results['total_bands'])
                            st.metric("Jan 2024 Images", results['jan_2024_sample_count'])
                        with col_b:
                            st.metric("SF Images", results['sf_sample_count'])
                            if 'sample_cloud_cover' in results:
                                st.metric("Sample Cloud Cover", f"{results['sample_cloud_cover']:.1f}%")
                        
                        st.write("**First 5 Bands:**", results['band_names'])
                        st.write("**Sample Image ID:**", results['first_image_id'])
                    else:
                        st.error(results['message'])
        
        with col3:
            if st.button("🗂️ Test Multiple Datasets"):
                with st.spinner("Testing multiple datasets..."):
                    results = test_multiple_datasets()
                    
                    st.write("### Dataset Access Results")
                    for name, result in results.items():
                        if result['success']:
                            st.success(f"{name}: ✅ Accessible")
                            with st.expander(f"Details for {name}"):
                                st.write(f"**Dataset ID:** {result['dataset_id']}")
                                st.write(f"**Sample Image:** {result['sample_image']}")
                        else:
                            st.error(f"{name}: ❌ Failed")
                            with st.expander(f"Error details for {name}"):
                                st.write(f"**Dataset ID:** {result['dataset_id']}")
                                st.write(f"**Error:** {result['error']}")

    elif analysis_mode == "📊 NDVI Point Analysis":
        st.header("📊 NDVI Point Analysis")
        st.write("Analyze vegetation health at a specific location")
        
        col1, col2 = st.columns(2)
        
        with col1:
            lat = st.number_input("Latitude", value=37.7749, format="%.6f", help="Latitude coordinate")
            lon = st.number_input("Longitude", value=-122.4194, format="%.6f", help="Longitude coordinate")
            
        with col2:
            start_date = st.date_input("Start Date", value=datetime(2023, 1, 1))
            end_date = st.date_input("End Date", value=datetime(2024, 12, 31))
        
        satellite = st.selectbox("Satellite", ["Sentinel-2", "Landsat-8"])
        
        if st.button("🚀 Analyze NDVI"):
            with st.spinner("Analyzing NDVI time series..."):
                result = analyze_ndvi_location(lat, lon, str(start_date), str(end_date), satellite)
                
                if result['success']:
                    df = result['data']
                    
                    st.success(f"✅ Analysis complete! Found {result['count']} data points.")
                    
                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean NDVI", f"{df['NDVI'].mean():.3f}")
                    with col2:
                        st.metric("Max NDVI", f"{df['NDVI'].max():.3f}")
                    with col3:
                        st.metric("Min NDVI", f"{df['NDVI'].min():.3f}")
                    with col4:
                        st.metric("Latest NDVI", f"{df['NDVI'].iloc[-1]:.3f}")
                    
                    # Interactive chart
                    chart_title = f"NDVI Time Series - {satellite} ({lat:.4f}, {lon:.4f})"
                    fig = create_interactive_ndvi_chart(df, chart_title)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Data table
                    if st.expander("📋 View Raw Data"):
                        st.dataframe(df)
                else:
                    st.error(f"❌ Analysis failed: {result['error']}")

    elif analysis_mode == "🗺️ NDVI Regional Analysis":
        st.header("🗺️ NDVI Regional Analysis")
        st.write("Analyze vegetation health for administrative regions")
        
        # Administrative level selection
        admin_level = st.selectbox("Administrative Level", 
                                   [0, 1, 2], 
                                   format_func=lambda x: {0: "Countries", 1: "States/Provinces", 2: "Municipalities"}[x])
        
        col1, col2 = st.columns(2)
        
        with col1:
            if admin_level == 0:
                # Country selection
                with st.spinner("Loading countries..."):
                    try:
                        countries_fc = get_admin_boundaries(0)
                        country_names = get_boundary_names(countries_fc, 0)
                        selected_country = st.selectbox("Select Country", country_names)
                        
                        if selected_country:
                            region_fc = countries_fc.filter(ee.Filter.eq('ADM0_NAME', selected_country))
                    except Exception as e:
                        st.error(f"Error loading countries: {str(e)}")
                        region_fc = None
            
            elif admin_level == 1:
                # State/Province selection
                with st.spinner("Loading countries..."):
                    try:
                        countries_fc = get_admin_boundaries(0)
                        country_names = get_boundary_names(countries_fc, 0)
                        selected_country = st.selectbox("Select Country", country_names)
                        
                        if selected_country:
                            # Get country code
                            country_info = countries_fc.filter(ee.Filter.eq('ADM0_NAME', selected_country)).first().getInfo()
                            country_code = country_info['properties']['ADM0_CODE']
                            
                            # Get states/provinces for this country
                            states_fc = get_admin_boundaries(1, country_code)
                            state_names = get_boundary_names(states_fc, 1)
                            selected_state = st.selectbox("Select State/Province", state_names)
                            
                            if selected_state:
                                region_fc = states_fc.filter(ee.Filter.eq('ADM1_NAME', selected_state))
                    except Exception as e:
                        st.error(f"Error loading administrative boundaries: {str(e)}")
                        region_fc = None
            
            else:  # admin_level == 2
                st.info("Municipality-level analysis available. Please select country and state first.")
                region_fc = None
        
        with col2:
            start_date = st.date_input("Start Date", value=datetime(2023, 1, 1), key="regional_start")
            end_date = st.date_input("End Date", value=datetime(2024, 12, 31), key="regional_end")
            satellite = st.selectbox("Satellite", ["Sentinel-2", "Landsat-8"], key="regional_satellite")
        
        if 'region_fc' in locals() and region_fc is not None:
            if st.button("🚀 Analyze Regional NDVI"):
                with st.spinner("Analyzing regional NDVI..."):
                    result = analyze_region_ndvi(region_fc, str(start_date), str(end_date), satellite)
                    
                    if result['success']:
                        df = result['data']
                        
                        st.success(f"✅ Regional analysis complete! Found {result['count']} data points.")
                        
                        # Statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean NDVI", f"{df['NDVI'].mean():.3f}")
                        with col2:
                            st.metric("Max NDVI", f"{df['NDVI'].max():.3f}")
                        with col3:
                            st.metric("Min NDVI", f"{df['NDVI'].min():.3f}")
                        with col4:
                            st.metric("Latest NDVI", f"{df['NDVI'].iloc[-1]:.3f}")
                        
                        # Interactive chart
                        region_name = selected_country if admin_level == 0 else f"{selected_state}, {selected_country}"
                        chart_title = f"Regional NDVI Time Series - {satellite} ({region_name})"
                        fig = create_interactive_ndvi_chart(df, chart_title)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Data table
                        if st.expander("📋 View Raw Data"):
                            st.dataframe(df)
                    else:
                        st.error(f"❌ Regional analysis failed: {result['error']}")

    elif analysis_mode == "🧪 Dataset Access Test":
        st.header("🧪 Dataset Access Test")
        st.write("Test access to various Google Earth Engine datasets")
        
        if st.button("🔬 Run Comprehensive Dataset Test"):
            with st.spinner("Testing access to multiple datasets..."):
                results = test_multiple_datasets()
                
                # Summary metrics
                total_datasets = len(results)
                successful_datasets = sum(1 for r in results.values() if r['success'])
                failed_datasets = total_datasets - successful_datasets
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Datasets", total_datasets)
                with col2:
                    st.metric("Accessible", successful_datasets, delta=None)
                with col3:
                    st.metric("Failed", failed_datasets, delta=None)
                
                # Detailed results
                st.write("### Detailed Results")
                
                # Successful datasets
                successful = [name for name, result in results.items() if result['success']]
                if successful:
                    st.success(f"✅ **Accessible Datasets ({len(successful)}):**")
                    for name in successful:
                        result = results[name]
                        with st.expander(f"{name} - ✅ Accessible"):
                            st.write(f"**Dataset ID:** `{result['dataset_id']}`")
                            st.write(f"**Sample Image:** `{result['sample_image']}`")
                
                # Failed datasets
                failed = [name for name, result in results.items() if not result['success']]
                if failed:
                    st.error(f"❌ **Failed Datasets ({len(failed)}):**")
                    for name in failed:
                        result = results[name]
                        with st.expander(f"{name} - ❌ Failed"):
                            st.write(f"**Dataset ID:** `{result['dataset_id']}`")
                            st.write(f"**Error:** {result['error']}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🌍 Google Earth Engine Dashboard | Built with Streamlit</p>
        <p>Powered by Google Earth Engine's petabyte-scale geospatial datasets</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Authentication failed - show setup instructions
    st.markdown("---")
    st.header("🔧 Setup Instructions")
    st.markdown("""
    To use this application, you need to configure your Google Earth Engine service account credentials in Replit Secrets:
    
    ### Steps to set up authentication:
    
    1. **Create a Google Cloud Project**
       - Go to [Google Cloud Console](https://console.cloud.google.com/)
       - Create a new project or select an existing one
    
    2. **Enable Earth Engine API**
       - In your Google Cloud project, enable the Earth Engine API
       - Go to APIs & Services → Library → Search for "Earth Engine API" → Enable
    
    3. **Create a Service Account**
       - Go to IAM & Admin → Service Accounts
       - Click "Create Service Account"
       - Give it a name and description
       - Grant it the "Earth Engine Resource Viewer" role (or Editor if you need write access)
    
    4. **Generate Service Account Key**
       - Click on your newly created service account
       - Go to the "Keys" tab
       - Click "Add Key" → "Create new key" → Select "JSON"
       - Download the JSON file
    
    5. **Add to Replit Secrets**
       - In your Replit project, open the Secrets tab (🔒 icon in the sidebar)
       - Add a new secret with key: `GOOGLE_SERVICE_ACCOUNT_JSON`
       - Copy and paste the entire contents of your downloaded JSON file as the value
    
    6. **Restart the Application**
       - Click the "🔄 Retry Authentication" button above
    
    ### Security Note:
    Your service account credentials are stored securely in Replit Secrets and are not visible in your code or to other users.
    """)
    
    st.info("💡 Need help? Check the [Google Earth Engine documentation](https://developers.google.com/earth-engine/guides/service_account) for detailed setup instructions.")
