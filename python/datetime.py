import pandas as pd

def string_to_date(pdf: pd.DataFrame, date_var: str) -> pd.Series:

  """
  Converts a string date column having 'YYYY-MM-DD' to datetime column

  Args:
      pdf (pd.DataFrame): Input pandas dataframe
      date_var (str): Date column in string format

  Returns:
      pd.Series: A pandas series in datetime type
  """
  return pdf[date_var].apply(lambda x: pd.to_datetime(x,errors='coerce',format='%Y-%m-%d'))

def date_to_month(pdf: pd.DataFrame, date_var: str) -> pd.Series:

  """
  Converts a datetime column to 'YYYYMM' string format

  Args:
      pdf (pd.DataFrame): Input pandas dataframe
      date_var (str): The datetime column to be converted

  Returns:
      pd.Series: A pandas series in string type

  """
  return pdf[date_var].apply(lambda x: x.strftime('%Y%m') if x is not pd.NaT else x)


def clean_date(pdf: pd.DataFrame, date_var: str) -> pd.Series:
  """
  Clean a date column when imported from a csv in string format

  Args:
      pdf (pd.DataFrame): Input pandas dataframe
      date_var (str): Date column in string format

  Returns:
      pd.Series: A pandas series in datetime type
  """
  return pdf[date_var].apply(lambda x: x[0:10])