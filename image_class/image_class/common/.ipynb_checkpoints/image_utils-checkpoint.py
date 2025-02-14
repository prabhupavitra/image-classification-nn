import logging
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from datetime import datetime 
import re

# Configure loggging for better traceability
logging.basicConfig(level=logging.INFO)

#  Function to extract label from a filename
def extract_label(filename):
    label = re.sub(r'\d+', '', filename)
    return label