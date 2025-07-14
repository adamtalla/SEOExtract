# SEO Keyword Extractor AI

A Flask-based web application that extracts the top 10 SEO keywords from any webpage using the YAKE (Yet Another Keyword Extractor) algorithm.

## Features

- **AI-Powered Analysis**: Uses YAKE algorithm for accurate keyword extraction
- **Clean Web Interface**: Bootstrap dark theme with responsive design
- **Dual API Support**: Both web form and JSON API endpoints
- **Real-time Processing**: Instant keyword extraction from any public URL
- **Copy to Clipboard**: Easy keyword copying functionality
- **Error Handling**: Comprehensive error handling and user feedback
- **No Database Required**: Stateless application, easy to deploy

## Quick Start

### Local Development

1. **Clone or Download** all the project files to your local machine
2. **Install Python 3.11+** if not already installed
3. **Install dependencies**:
   ```bash
   pip install flask flask-cors trafilatura yake requests gunicorn
   ```
4. **Run the application**:
   ```bash
   python main.py
   ```
5. **Open your browser** and go to `http://localhost:5000`

### Production Deployment

#### Using Gunicorn (Recommended)

```bash
gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app
```

#### Using Docker

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--reuse-port", "main:app"]
```

Create a `requirements.txt`:
```
flask
flask-cors
trafilatura
yake
requests
gunicorn
```

Build and run:
```bash
docker build -t seo-keyword-extractor .
docker run -p 5000:5000 seo-keyword-extractor
```

## File Structure

```
seo-keyword-extractor/
├── app.py                 # Main Flask application
├── main.py               # Application entry point
├── keyword_extractor.py  # YAKE keyword extraction logic
├── web_scraper.py        # Web scraping functionality
├── templates/
│   └── index.html        # Main web interface
├── static/
│   ├── style.css         # Custom styling
│   └── script.js         # Frontend JavaScript
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## API Endpoints

### Web Interface
- `GET /` - Main application interface
- `POST /extract` - Extract keywords via web form

### JSON API
- `POST /api/extract_keywords` - Extract keywords via JSON API

#### API Usage Example

```bash
curl -X POST http://localhost:5000/api/extract_keywords \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

Response:
```json
{
  "keywords": [
    "example keyword 1",
    "keyword 2",
    "... up to 10 keywords"
  ],
  "url": "https://example.com"
}
```

## Environment Variables

- `SESSION_SECRET` - Secret key for Flask sessions (optional, defaults to "dev-secret-key")

## Hosting Options

### 1. Heroku
1. Create a `Procfile`:
   ```
   web: gunicorn --bind 0.0.0.0:$PORT main:app
   ```
2. Deploy to Heroku using Git

### 2. Railway
1. Connect your GitHub repository
2. Railway will automatically detect and deploy

### 3. Vercel
1. Create a `vercel.json`:
   ```json
   {
     "builds": [
       {
         "src": "main.py",
         "use": "@vercel/python"
       }
     ],
     "routes": [
       {
         "src": "/(.*)",
         "dest": "main.py"
       }
     ]
   }
   ```

### 4. DigitalOcean App Platform
1. Create an app from your GitHub repository
2. Configure build and run commands in the UI

### 5. AWS/GCP/Azure
Deploy using their respective container services or serverless functions.

## Dependencies

- **Flask** - Web framework
- **Flask-CORS** - Cross-origin resource sharing
- **Trafilatura** - Web content extraction
- **YAKE** - Keyword extraction algorithm
- **Requests** - HTTP library
- **Gunicorn** - WSGI HTTP server

## How It Works

1. **URL Input**: User enters a website URL
2. **Content Extraction**: Trafilatura extracts clean text content
3. **Keyword Analysis**: YAKE algorithm identifies top keywords
4. **Results Display**: Keywords are ranked and displayed
5. **Copy Function**: Users can copy keywords to clipboard

## Technical Details

- **Language**: Python 3.11+
- **Framework**: Flask
- **Algorithm**: YAKE (Yet Another Keyword Extractor)
- **Scraping**: Trafilatura library
- **Frontend**: Bootstrap 5 with dark theme
- **Deployment**: Gunicorn WSGI server

## Security Notes

- The application doesn't store any user data
- All processing happens in memory
- No authentication required
- CORS enabled for API access

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Port Conflicts**: Change port in `main.py` if needed
3. **Permission Errors**: Check file permissions
4. **Network Issues**: Ensure outbound HTTP/HTTPS access

### Debug Mode

Enable debug mode by setting `debug=True` in `main.py` for development.

## License

MIT License - feel free to use and modify as needed.

## Support

For issues or questions, check the console logs for detailed error messages.