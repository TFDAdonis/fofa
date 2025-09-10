# Overview

This is a Streamlit-based web application that provides a comprehensive Google Earth Engine (GEE) dashboard for satellite data analysis and visualization. The application focuses on NDVI (Normalized Difference Vegetation Index) analysis using Sentinel-2 and Landsat satellite imagery, featuring interactive forex-style charts and time-series vegetation monitoring capabilities.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit - chosen for rapid development of data science web applications with minimal frontend complexity
- **Layout**: Wide layout with expandable sidebar for better data visualization experience
- **Visualization**: Plotly for interactive charts with forex-style time-series plots and candlestick-style visualizations
- **User Interface**: Form-based input with real-time feedback, file upload capabilities, and dashboard-style layout

## Backend Architecture
- **Runtime**: Python-based application running on Streamlit's built-in server
- **Authentication Flow**: Service account-based authentication using Google OAuth2 credentials stored in Replit Secrets
- **Data Processing Pipeline**: Multi-stage processing including cloud masking, NDVI calculation, and time-series analysis
- **Error Handling**: Comprehensive try-catch blocks with user-friendly error messages and authentication state management

## Data Processing Architecture
- **Satellite Data Access**: Integration with Google Earth Engine's Python API for accessing Sentinel-2 and Landsat collections
- **Image Processing**: Cloud masking functions for Sentinel-2 imagery using QA60 band quality assessment
- **NDVI Calculation**: Automated vegetation index computation using Near-Infrared and Red spectral bands
- **Time-Series Analysis**: Temporal aggregation and trend analysis for vegetation monitoring
- **Data Export**: Pandas DataFrame integration for statistical analysis and CSV export capabilities

## Visualization System
- **Interactive Charts**: Plotly-based interactive time-series plots with zoom, pan, and hover capabilities
- **Chart Types**: Line plots, candlestick charts, and multi-subplot layouts for comprehensive data exploration
- **Real-time Updates**: Dynamic chart updates based on user selections and parameter changes
- **Data Presentation**: Statistical summaries, trend analysis, and comparative visualizations

## Security Architecture
- **Authentication Method**: Google service account credentials with Earth Engine scope permissions
- **Credential Storage**: Environment variable-based storage using Replit Secrets for secure credential management
- **API Scoping**: Restricted to Earth Engine API access with appropriate OAuth2 scopes
- **Session Management**: Streamlit session state for maintaining authentication status across user interactions

# External Dependencies

## Google Services
- **Google Earth Engine API**: Primary service for satellite imagery and geospatial data access
- **Google OAuth2**: Authentication service for service account credential validation
- **Earth Engine Python Client Library**: Official Python SDK for GEE integration
- **Earth Engine Data Catalog**: Access to Sentinel-2, Landsat, and other satellite imagery collections

## Python Libraries
- **Streamlit**: Web application framework for the dashboard interface
- **google-auth**: Google authentication library for handling service account credentials
- **earthengine-api**: Google Earth Engine Python API client
- **Plotly**: Interactive visualization library for charts and graphs
- **Pandas**: Data manipulation and analysis for time-series processing
- **NumPy**: Numerical computing for array operations and mathematical functions
- **Matplotlib**: Additional plotting capabilities for static visualizations

## Satellite Data Sources
- **Sentinel-2 Collection**: Primary dataset for high-resolution multispectral imagery
- **Landsat Collections**: Secondary dataset for long-term historical analysis
- **Quality Assessment Bands**: Cloud masking and quality control using satellite metadata
- **Temporal Collections**: Time-series datasets for vegetation trend analysis

## Environment Configuration
- **Replit Secrets**: Secure storage for Google service account credentials
- **Environment Variables**: Configuration management for API keys and authentication
- **Session State**: Streamlit's built-in state management for user authentication persistence