# Clustering Map Backend

This is the backend API for Clustering Map, built with FastAPI and Python.

## Overview

Clustering Map is a web application that analyzes Excel survey data and generates clustering maps for data visualization. This backend provides REST APIs for file processing, data analysis, and result export.

## Features

- **Excel File Processing**: Upload and process Excel survey data
- **Text Analysis**: Extract and analyze text content using TF-IDF
- **Clustering**: Generate clusters using KMeans algorithm
- **Data Export**: Export results as PDF or PNG
- **Configuration Management**: Save and load analysis configurations
- **Result Management**: Save and retrieve analysis results

## Technology Stack

- **Backend**: FastAPI + Python 3.11
- **Data Processing**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn, plotly
- **Export**: reportlab, pillow
- **Deployment**: Render (Docker)

## API Endpoints

### Health Check
- `GET /` - Root endpoint
- `GET /health` - Health check

### File Upload
- `POST /upload` - Upload Excel file and get column mapping

### Analysis
- `POST /analyze` - Analyze data and generate clustering results
- `GET /configs` - Get saved configurations
- `POST /configs` - Save configuration
- `GET /results` - Get saved results
- `POST /results` - Save results

### Export
- `GET /export/pdf` - Export results as PDF
- `GET /export/png` - Export results as PNG

## Development

### Prerequisites

- Python 3.11 or higher
- pip

### Installation

```bash
pip install -r requirements-minimal.txt
```

### Development Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing

```bash
pytest app/tests/
```

## Environment Variables

- `PYTHONPATH`: Python path (default: /app)
- `ENVIRONMENT`: Environment (production/development)

## Deployment

This application is deployed on Render as a Web Service using Docker.

### Manual Deployment

```bash
git add .
git commit -m "Update backend"
git push origin main
```

## Project Structure

```
clustering-map-backend/
├── app/
│   ├── models/         # Pydantic models
│   ├── services/       # Business logic services
│   ├── utils/          # Utility functions
│   ├── tests/          # Test files
│   └── main.py         # FastAPI application
├── requirements-minimal.txt  # Python dependencies
├── Dockerfile.render   # Docker configuration
└── render.yaml         # Render deployment configuration
```

## API Integration

The backend provides APIs for the Clustering Map frontend:

- File upload and processing
- Data analysis and clustering
- Result visualization data
- Export functionality
- Configuration management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License