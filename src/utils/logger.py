"""
Enhanced logger utility with colors and better formatting.
"""
import logging
import sys

# ANSI color codes
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Levels
    DEBUG = '\033[36m'      # Cyan
    INFO = '\033[32m'       # Green
    WARNING = '\033[33m'    # Yellow
    ERROR = '\033[31m'      # Red
    CRITICAL = '\033[35m'   # Magenta
    
    # Custom
    SUCCESS = '\033[92m'    # Bright green
    HIGHLIGHT = '\033[96m'  # Bright cyan


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors."""
    
    FORMATS = {
        logging.DEBUG: f"{Colors.DEBUG}%(asctime)s - %(name)s - DEBUG - %(message)s{Colors.RESET}",
        logging.INFO: f"{Colors.INFO}%(asctime)s - %(name)s - INFO - %(message)s{Colors.RESET}",
        logging.WARNING: f"{Colors.WARNING}%(asctime)s - %(name)s - WARNING - %(message)s{Colors.RESET}",
        logging.ERROR: f"{Colors.ERROR}%(asctime)s - %(name)s - ERROR - %(message)s{Colors.RESET}",
        logging.CRITICAL: f"{Colors.CRITICAL}%(asctime)s - %(name)s - CRITICAL - %(message)s{Colors.RESET}"
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


def get_logger(name: str, level: int = logging.INFO, use_colors: bool = True, log_file: str = None):
    """
    Get configured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level
        use_colors: Whether to use colored output
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        if use_colors and sys.stdout.isatty():
            console_formatter = ColoredFormatter()
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            # Create logs directory if needed
            import os
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Overwrite mode - fresh log each run
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        logger.setLevel(level)
        logger.propagate = False  # Prevent duplicate logs
    
    return logger


def log_dict(logger, data: dict, title: str = None, indent: int = 2):
    """
    Log dictionary in clean format.
    
    Args:
        logger: Logger instance
        data: Dictionary to log
        title: Optional title
        indent: Indentation level
    """
    if title:
        logger.info(f"\n{title}")
        logger.info("=" * len(title))
    
    for key, value in data.items():
        if isinstance(value, dict):
            logger.info(f"{' ' * indent}{key}:")
            log_dict(logger, value, indent=indent+2)
        elif isinstance(value, list):
            logger.info(f"{' ' * indent}{key}: {value if len(value) <= 3 else value[:3] + ['...']}")
        else:
            logger.info(f"{' ' * indent}{key}: {value}")


def log_separator(logger, char: str = "=", length: int = 60):
    """Log a separator line."""
    logger.info(char * length)
