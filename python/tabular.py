import pandas as pd

def transpose_table(pdf: pd.DataFrame, var: str, row_var: list , column_var: str, function: str, rename_cols=True) -> pd.DataFrame:

  """
  Gets the transpose(pivot) of a dataframe with option to rename the resultant columns

  Args:
      pdf (pd.DataFrame): Input pandas dataframe
      var (str): The column whose values will be calculated
      row_var (list): List of key columns
      column_var (str): The column whose distinct values will comprise the columns
      function (str): The aggregate function
      rename_cols (bool): Option to rename the column, default=True

  Returns:
      pd.DataFrame: A transformed dataframe with the first columns as columns specified in the row_var list and the other columns belonging to the categories of the column_var
  """
  df = pd.pivot_table(pdf, values= var, index= row_var, columns= column_var, aggfunc= function).reset_index()
  if rename_cols==True:
    return df.rename(columns={c: var+'_'+c for c in df.columns if c not in row_var})
  else:
    return df
  
def agg_cols(pdf: pd.DataFrame, col_list: list, func: str, skipna: bool = True) -> pd.Series:
  """
  Aggregate a list of columns

  Args:
      pdf (pd.DataFrame): Input pandas dataframe
      col_list (list): List of columns to aggregate
      func (str): The aggregate function
      skipna (bool): skip NA, default = True
  
  Returns:
     pd.Series: A pandas series with values which is the sum along the columns

  Raise:
      ValueError: When func is not any of sum, avg, max, min
  """

  if func == 'sum':
    return pdf[col_list].sum(axis=1, skipna = skipna)
  elif func == 'avg':
    return pdf[col_list].mean(axis=1, skipna = skipna)
  elif func == 'max':
    return pdf[col_list].max(axis=1, skipna = skipna)
  elif func == 'min':
    return pdf[col_list].min(axis=1, skipna = skipna)
  else:
    raise ValueError('func only takes values: sum, avg, max, min')
  
def agg_cols_pos(pdf: pd.DataFrame, string: str, position: str, func: str) -> pd.Series:
  """
  Aggregate by columns where the columns have a common string

  Args:
      pdf (pd.DataFrame): Input pandas dataframe
      string (str): The common string to select columns
      position (str): The position of the string to search
      func (str): The aggregate function
  
  Returns:
     pd.Series: A pandas series with values which is the sum along the columns

  Raise:
      ValueError: When the position is not any of starts, ends, contains
      ValueError: When func is not any of sum, avg, max, min
  """
  if position == 'starts':
    l = [c for c in pdf.columns if c.startswith(string)]
  elif position == 'ends':
    l = [c for c in pdf.columns if c.endswith(string)]
  elif position == 'contains':
    l = [c for c in pdf.columns if string in c]
  else:
    raise ValueError('positiion only takes values: starts, ends, contains')

  if func == 'sum':
    return pdf[l].sum(axis=1)
  elif func == 'avg':
    return pdf[l].mean(axis=1)
  elif func == 'max':
    return pdf[l].max(axis=1)
  elif func == 'min':
    return pdf[l].min(axis=1)
  else:
    raise ValueError('func only takes values: sum, avg, max, min')

def rolling_aggregate(pdf: pd.DataFrame, var: str, n: int, func: str, key_var: str, time_var: str) -> pd.DataFrame:

  """
    Get rolling aggreagte of a numeric column based on a key and time variable.
    It will calculate the rolling aggregate considering values of the last n rows

    Args:
        pdf (pd.DataFrame): Input pandas dataframe.
        var (str): Column to aggragate
        n (int): Number of periods to be considered for calculating the aggregate
        func (str): Aggregate function
        key_var (str): Key column
        time_var (str): The time variable

    Returns:
        pd.DataFrame: The original DataFrame adding the output aggregate column

    Raises:
        ValueError: When the func is not any of the values: sum, avg, max, min, med, std

  """

  t = pdf.sort_values([key_var, time_var], ascending=[True, True])
  if func == 'sum':
    t[func+'_'+var+'_'+str(n)] = t.groupby(key_var)[var].rolling(window=n,min_periods=1).sum().reset_index()[var].values
  elif func == 'max':
    t[func+'_'+var+'_'+str(n)] = t.groupby(key_var)[var].rolling(window=n,min_periods=1).max().reset_index()[var].values
  elif func == 'min':
    t[func+'_'+var+'_'+str(n)] = t.groupby(key_var)[var].rolling(window=n,min_periods=1).min().reset_index()[var].values
  elif func == 'med':
    t[func+'_'+var+'_'+str(n)] = t.groupby(key_var)[var].rolling(window=n,min_periods=1).median().reset_index()[var].values
  elif func == 'std':
    t[func+'_'+var+'_'+str(n)] = t.groupby(key_var)[var].rolling(window=n,min_periods=1).std().reset_index()[var].values
  elif func == 'avg':
    t[func+'_'+var+'_'+str(n)] = t.groupby(key_var)[var].rolling(window=n,min_periods=1).mean().reset_index()[var].values
  else:
    raise ValueError('func only takes values: sum, avg, max, min, med, std')

  return t

