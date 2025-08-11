"""
Logging utilities for the hybridRAG system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from config.settings import get_settings

def setup_logging(
    level: str = None,
    log_file: Optional[str] = None,
    log_format: str = None
) -> logging.Logger:
    """Set up logging configuration."""
    
    # Get settings
    settings = get_settings()
    level = level or settings.log_level
    
    # Create logger
    logger = logging.getLogger('hybridRAG')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(log_format)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance."""
    if name:
        return logging.getLogger(f'hybridRAG.{name}')
    return logging.getLogger('hybridRAG')

def log_function_call(logger: logging.Logger, func_name: str, **kwargs):
    """Log function call with parameters."""
    params = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
    logger.debug(f"Calling {func_name}({params})")

def log_function_result(logger: logging.Logger, func_name: str, result, **kwargs):
    """Log function result."""
    if isinstance(result, (list, tuple)):
        result_summary = f"{type(result).__name__} with {len(result)} items"
    elif isinstance(result, dict):
        result_summary = f"{type(result).__name__} with {len(result)} keys"
    else:
        result_summary = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
    
    logger.debug(f"{func_name} returned: {result_summary}")

def log_error(logger: logging.Logger, error: Exception, context: str = ""):
    """Log error with context."""
    logger.error(f"Error in {context}: {str(error)}", exc_info=True)

def log_performance(logger: logging.Logger, operation: str, duration: float, **kwargs):
    """Log performance metrics."""
    extra_info = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"Performance: {operation} took {duration:.3f}s {extra_info}")

# Default logger instance
logger = get_logger()

# Convenience functions
def info(message: str):
    """Log info message."""
    logger.info(message)

def debug(message: str):
    """Log debug message."""
    logger.debug(message)

def warning(message: str):
    """Log warning message."""
    logger.warning(message)

def error(message: str):
    """Log error message."""
    logger.error(message)

def critical(message: str):
    """Log critical message."""
    logger.critical(message)
