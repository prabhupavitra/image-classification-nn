import logging
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from datetime import datetime 

# Configure loggging for better traceability
logging.basicConfig(level=logging.INFO)

#  Function to save dataframe to a specific table in spark

def save_dataframe(spark, df, table_name: str = "cgda_tt_screen_ets_ty22"):
    """
    Save a dataframe to a specified spark table, dropping the existing table if it exists
    
    Parameters : 
    - spark (Spark Session) : The Spark Session
    - df (DataFrame) : The Dataframe to save
    - table_name (str) : The name of the table to save data info

    """

    # Log the action
    logging.info(f"Saving Dataframe to table {table_name}")

    # Drop the existing table 
    spark.sql(f"DROP TABLE IF EXISTS cgan_ustax_ws.{table_name}")

    try:
        # Write the dataframe to a parquet table with snappy compression
        df.write.format("parquet").mode("append").option("spark.sql.parquet.compression.codex","snappy").saveAsTable(f"cgan_ustax_ws.{table_name}")
    except Exception as e:
        logging.error(f" Error saving dataframe to {table_name} : {e}")


def save_dataframe_partition(
        spark ,
        df,
        partition: str,
        partitions : list = ["partition_date"],
        table_name : str = "cgda_tt_screen_ets_ty22",
):
    """
    Save a dataframe to a specified spark table, dropping the existing table if it exists
    
    Parameters : 
    - spark (Spark Session) : The Spark Session
    - df (DataFrame) : The Dataframe to save
    - partition (str) : The specific partition value to save
    - partitions (list) : List of partition columns
    - table_name (str) : The name of the table to save data info

    """

    # Log details of saving partitioned dataframe
    logging.info(f"Saving datadrame partition {partition} to {table_name}")
    # logging.info(f"Count inserted while saving dataframe : {df.count()}")

    # Check if the table exists before dropping
    msg = spark.catalog.tableExists(f"cgan_ustax_ws.{table_name}")
    logging.info(f"Table exists : cgan_ustax_ws.{table_name} - {msg}")

    # Append data to existing table
    try:
        df.write.format("parquet").mode("append").option("spark.sql.parquet.compression.codex","snappy").partitionBy(partitions[0]).saveAsTable(f"cgan_ustax_ws.{table_name}")
    # Refresh table if neeeded
    except Exception as e:
        logging.error(f"Error saving data frame partition to {table_name} : {e}")

    
def drop_partition(
        spark,
        table_name : str,
        partition_name : list = ["partition_date"],
        partition_value : str = "2022-01-01",
):
    """
    Drop a specific partition from a given spark table

    Parameters:
    - spark
    - table_name
    - partition_name : List of partition column names
    - partition_value : The specific value of the partition to drop
    
    """
    # Log the action of dropping the partition
    logging.info(
        f" Dropping partition {partition_name} = '{partition_value}' from {table_name}"
    )

    if spark.catalog.tableExists(f"cgan_ustax_ws.{table_name}"):
        # SQL to drop a specific partition
        sql_query = f""" ALTER TABLE cgan_ustax_ws.{table_name} DROP IF EXISTS PARTITION ({partition_name[0]} = '{partition_value}')
        """
        logging.info(f"Executing sql : {sql_query}")

        try :
            spark.sql(sql_query)
            logging.info("Partition dropped successfully")
        except Exception as e:
            logging.error(f" Error dropping partition {partition_value} : {e}")
    else:
        logging.warning(f" Table {table_name} does not exist and cannot drop partition")


def get_partition_date_list(spark, tax_year : int,start_date : str, end_date: str)-> list :
    """
    Retrieves a list of partition dates depending on tax year and specidied data range
    """
    # Log the retrieval of partition dates
    logging.info(f" Getting partition date list for TY {tax_year} from {start_date} to {end_date}")

    #  SQL QUERY TO EXTRACT PARTITION DATES
    extract_partition_sql = f"""
    SELECT CAST(tax_utc_date as string) as partition_date
    from common_dm.dim_cg_date
    where tax_year = {tax_year}
    and tax_utc_date between "{start_date}" and "{end_date}"
    GROUP BY 1 ORDER BY 1 ASC
    """

    #  Execute the sql query and return the list of dates
    partition_df = spark.sql(extract_partition_sql)
    return [row.partition_date for row in partition_df.select["partition_date"].collect()]


def get_partition_months(partition_date : str) -> list:

    # Convert partition date to a datetime object
    vpartitiondate = datetime.strptime(partition_date,"%Y-%m-%d")

    # Getting the first day of current month
    first_day_current = vpartitiondate.replace(day=1)

    # Getting first day of next month
    next_month = first_day_current + pd.offsets.MonthBegin(n=1)
    
    return [first_day_current.strftime("%m"), next_month.strftime("%m")]

    
