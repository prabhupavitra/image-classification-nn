import logging
import pandas as pd
import re as re
from datetime import datetime 

# Configure loggging for better traceability
logging.basicConfig(level=logging.INFO)

# Function to extract label from a filename
def extract_label(filename):
    # Check if the filename has an extension
    if '.' not in filename:
        raise ValueError("Filename must have an extension.")
    
    # Extract the file extension
    extension = filename.split('.')[-1]

    # Remove numbers and dots, then remove the extension
    var = re.sub(r'\.|\d+', '', filename.split('.')[0])

    # Get the label (the last part of the path)
    label = var.split('/')[-1]
    
    return label


    