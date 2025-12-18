import uvicorn
from config.logging_config import setup_logging

if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    # Run the FastAPI application
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=None  # Use our custom logging config
    )