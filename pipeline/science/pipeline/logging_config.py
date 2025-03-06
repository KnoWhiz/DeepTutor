import sys

import logging
logger = logging.getLogger("tutorpipeline.science.logging_config")

def setup_logging():
    # Create a formatter that includes timestamp, level, and logger name
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create console handler and set formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Get the root logger and set its level
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add the console handler to the root logger
    root_logger.addHandler(console_handler)

    # Set up specific loggers for our application
    tutorpipeline_logger = logging.getLogger("tutorpipeline")
    tutorpipeline_logger.setLevel(logging.INFO) 