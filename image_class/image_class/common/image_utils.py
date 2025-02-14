import logging
import pandas as pd
# from pyspark.sql import SparkSession
# from pyspark.sql.types import IntegerType
# from datetime import datetime 
# import re

# Configure loggging for better traceability
logging.basicConfig(level=logging.INFO)

# #  Function to extract label from a filename
# def extract_label(filename):
#     label = re.sub(r'\d+', '', filename)
#     return label

#  Function to extract label from a filename
def extract_label(filename):
    extension = filename.split('.')[-1]
    var = re.sub(extension,'',re.sub(r'\.|\d+', '', filename))
    label = var.split('/')[-1]
    return label

#  Function to extract label from a filename
def cleanup_dataset(filename):
    extension = filename.split('.')[-1]
    var = re.sub(extension,'',re.sub(r'\.|\d+', '', filename))
    label = var.split('/')[-1]
    return label