import pyspark
import pyspark.sql.functions as F 

def clean_date(pdf: pyspark.sql.dataframe.DataFrame, date_col_list: list) -> pyspark.sql.dataframe.DataFrame:
  """
    Clean a date column in string format and convert into date format

    Args:
        pdf (pyspark.sql.dataframe.DataFrame): Input pyspark sql dataframe.
        date_col_list (list): List of date columns

    Returns:
        pyspark.sql.dataframe.DataFrame: The original DataFrame transforming the date columns to from string to date format.
  """
  for i in date_col_list:
     pdf = pdf.withColumn(i, F.to_date(F.substring(F.col(i),1,10),'yyyy-MM-dd'))
  return pdf

def first_date_month(pdf: pyspark.sql.dataframe.DataFrame, date_var: str, output_date_var: str) -> pyspark.sql.dataframe.DataFrame:
  """
    Get the starting month date of the respective dates in a date column

    Args:
        pdf (pyspark.sql.dataframe.DataFrame): Input pyspark sql dataframe.
        date_var (str): Input date column
        output_date_var (str): Output column of starting month dates of respective date_var values

    Returns:
        pyspark.sql.dataframe.DataFrame: The original DataFrame adding the output_date_var
  """
  return pdf.withColumn(output_date_var, F.trunc(date_var, "month"))

def end_date_month(pdf: pyspark.sql.dataframe.DataFrame, output_date_var: str, input_date_var: str) -> pyspark.sql.dataframe.DataFrame:
  """
  For an input date column it provides the respective last date of the month

  Args:
     pdf (pyspark.sql.dataframe.DataFrame): Input dataframe
     output_date_var (str): the output column name
     input_date_var (str) : The input date column name

  Returns:
      pyspark.sql.dataframe.DataFrame: The original dataframe with the output_date_var column added

  """

  return pdf.withColumn(output_date_var, F.last_day(F.col(input_date_var)))

def date_to_month(pdf: pyspark.sql.dataframe.DataFrame, output_month: str, input_date: str) -> pyspark.sql.dataframe.DataFrame:

  """
  Converts a date column ( in 'YYYY-MM-DD' format) to 'YYYYMM' long format

  Args:
      pdf (pyspark.sql.dataframe.DataFrame): Input dataframe
      output_month (str): The output long format column name
      input_date (str): The input date format column name

  Returns:
      pyspark.sql.dataframe.DataFrame: The original dataframe with the output_month column added

  """
  return pdf.withColumn(output_month, F.substring(F.regexp_replace(F.col(input_date), '-', ''),1,6).cast('long'))