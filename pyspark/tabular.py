import pyspark
import pyspark.sql.functions as F 

def agg_cols(pdf: pyspark.sql.dataframe.DataFrame, output_var: str, col_list: list, func: str) -> pyspark.sql.dataframe.DataFrame:
  """
    Aggregate multiple columns at a row level 

    Args:
        pdf (pyspark.sql.dataframe.DataFrame): Input pyspark sql dataframe.
        output_var (str): Name of the output column
        col_list (list): List of input columns
        metric (str): 'sum', 'avg', 'max' or 'min', the aggregate function applied across columns

    Returns:
        pyspark.sql.dataframe.DataFrame: The original DataFrame adding the output_var.

    Raises:
        ValueError: When the func is not any of the values: sum, avg, min, max
  """
  if func == 'sum':
    return pdf.withColumn(output_var, sum([F.col(x) for x in col_list]))
  elif func == 'avg':
    return pdf.withColumn(output_var, sum([F.col(x) for x in col_list])/len(col_list))
  elif func == 'max':
    return pdf.withColumn(output_var, F.greatest(*col_list))
  elif func == 'min':
    return pdf.withColumn(output_var, F.least(*col_list))
  else:
    raise ValueError("func only takes values sum, avg, min, max")


def get_ratio(pdf: pyspark.sql.dataframe.DataFrame, output_var: str, numerator_var: str, denominator_var: str, replacement_value: float) -> pyspark.sql.dataframe.DataFrame:

  """
    Get the ratio of two columns and assign a float value when the denominator is 0 or Null

    Args:
        pdf (pyspark.sql.dataframe.DataFrame): Input pyspark sql dataframe.
        output_var (str): Name of the output column with the ratio
        numerator_var (str): Numerator column
        denominator_var (str): Denominator column
        replacement_value (float): Float value to replace with when denominator is Null or 0

    Returns:
        pyspark.sql.dataframe.DataFrame: The original DataFrame adding the output_var

  """
  return pdf.withColumn(output_var, F.when(F.col(denominator_var)==0, replacement_value).otherwise(F.col(numerator_var)/F.col(denominator_var)))

def rolling_aggregate(pdf: pyspark.sql.dataframe.DataFrame,var: str,n: int,func: str,key_var: str,time_var: str) -> pyspark.sql.dataframe.DataFrame:

  """
    Get rolling aggreagte of a numeric column based on a key and time variable.
    It will calculate the rolling aggregate considering values of the last n rows

    Args:
        pdf (pyspark.sql.dataframe.DataFrame): Input pyspark sql dataframe.
        var (str): Column to aggragate
        n (int): Number of periods to be considered for calculating the aggregate
        func (str): Aggregate function
        key_var (str): Key column
        time_var (str): The time variable

    Returns:
        pyspark.sql.dataframe.DataFrame: The original DataFrame adding the output aggregate column

    Raises:
        ValueError: When the func is not any of the values: avg, min, max, med, std

  """
  from pyspark.sql.window import Window
  w = Window().partitionBy(key_var).orderBy(time_var).rowsBetween(-n+1, 0)
  if func == 'avg':
    return pdf.withColumn(f'{func}_{var}_{n}', F.avg(var).over(w))
  elif func == 'max':
    return pdf.withColumn(f'{func}_{var}_{n}', F.max(var).over(w))
  elif func == 'min':
    return pdf.withColumn(f'{func}_{var}_{n}', F.min(var).over(w))
  elif func == 'med':
    return pdf.withColumn(f'{func}_{var}_{n}', F.percentile_approx(var, 0.5, accuracy=1000000).over(w))
  elif func == 'std':
    return pdf.withColumn(f'{func}_{var}_{n}', F.std(var).over(w))
  elif func == 'sum':
    return pdf.withColumn(f'{func}_{var}_{n}', F.sum(var).over(w))
  else:
    raise ValueError("func only takes values: sum, avg, min, max, med, std")


def rolling_aggregate_pre(pdf: pyspark.sql.dataframe.DataFrame,var: str,n: int,func: str,key_var: str,time_var: str) -> pyspark.sql.dataframe.DataFrame:

  """
    Get rolling aggreagte of a numeric column based on a key and time variable for pre n period.

    For example, if n=3 then for row 6 it will calculate the aggregate from row 3 to row 1

    Args:
        pdf (pyspark.sql.dataframe.DataFrame): Input pysopark sql dataframe.
        var (str): Column to aggragate
        n (int): Number of periods to be considered for calculating the aggregate
        func (str): Aggregate function
        key_var (str): Key column
        time_var (str): Time column

    Returns:
        pyspark.sql.dataframe.DataFrame: The original DataFrame adding the output aggregate column

    Raises:
        ValueError: When the func is not any of the values: avg, min, max, med, std

  """
  from pyspark.sql.window import Window
  w = Window().partitionBy(key_var).orderBy(time_var).rowsBetween(-2*n+1, -n)
  if func == 'avg':
    return pdf.withColumn(f'{func}_{var}_{n}_pre', F.avg(var).over(w))
  elif func == 'max':
    return pdf.withColumn(f'{func}_{var}_{n}_pre', F.max(var).over(w))
  elif func == 'min':
    return pdf.withColumn(f'{func}_{var}_{n}_pre', F.min(var).over(w))
  elif func == 'med':
    return pdf.withColumn(f'{func}_{var}_{n}_pre', F.percentile_approx(var, 0.5, accuracy=1000000).over(w))
  elif func == 'std':
    return pdf.withColumn(f'{func}_{var}_{n}_pre', F.std(var).over(w))
  elif func == 'sum':
    return pdf.withColumn(f'{func}_{var}_{n}_pre', F.sum(var).over(w))
  else:
    raise ValueError("func only takes values: avg, min, max, med, std")

def rolling_aggregate_dynm(pdf: pyspark.sql.dataframe.DataFrame,var: str,n: int, k: int,func: str,key_var: str,time_var: str) -> pyspark.sql.dataframe.DataFrame:

  """
    Get rolling aggreagte of a numeric column based on a key and time variable for last n period with the option of setting the starting point of the period

    For example, if n=3 and k=1 then for row 6 it will calculate the aggregate from row 5 to row 3. The window starts from row 5 (as k=1) and takes the last 3 rows from the start of the window (n=3)

    Args:
        pdf (pyspark.sql.dataframe.DataFrame): Input pysopark sql dataframe.
        var (str): Column to aggragate
        n (int): Number of periods to be considered for calculating the aggregate
        k (int): Starting point of the period
        func (str): Aggregate function
        key_var (str): Key column
        time_var (str): Time column

    Returns:
        pyspark.sql.dataframe.DataFrame: The original DataFrame adding the output aggregate column

    Raises:
        ValueError: When the func is not any of the values: avg, min, max, med, std

  """
  from pyspark.sql.window import Window
  w = Window().partitionBy(key_var).orderBy(time_var).rowsBetween(-n-k+1, -k)
  if func == 'avg':
    return pdf.withColumn(f'{func}_{var}_{n}_pre_{k}', F.avg(var).over(w))
  elif func == 'max':
    return pdf.withColumn(f'{func}_{var}_{n}_pre_{k}', F.max(var).over(w))
  elif func == 'min':
    return pdf.withColumn(f'{func}_{var}_{n}_pre_{k}', F.min(var).over(w))
  elif func == 'med':
    return pdf.withColumn(f'{func}_{var}_{n}_pre_{k}', F.percentile_approx(var, 0.5, accuracy=1000000).over(w))
  elif func == 'std':
    return pdf.withColumn(f'{func}_{var}_{n}_pre_{k}', F.std(var).over(w))
  elif func == 'sum':
    return pdf.withColumn(f'{func}_{var}_{n}_pre_{k}', F.sum(var).over(w))
  else:
    raise ValueError("func only takes values: avg, min, max, med, std")


def trend_coeff(pdf: pyspark.sql.dataframe.DataFrame, var: str, n: int, key_var: str,time_var: str) -> pyspark.sql.dataframe.DataFrame:

  """
    Slope coefficient for the best fit line with intercept of a variable

    Args:
        pdf (pyspark.sql.dataframe.DataFrame): Input pyspark sql dataframe.
        var (str): Input variable
        n (int): Number of periods
        key_var (str): Key column
        time_var (str): Time column

    Returns:
        pyspark.sql.dataframe.DataFrame: The original DataFrame adding the output trend coefficient column

  """
  from pyspark.sql.window import Window
  from pyspark.sql.types import DoubleType

  # Create x_bar which is the average of the summation of n
  x_bar = (n*(n+1)/2)/n
  x_list = [i+1 for i in range(0,n)]
  # List containing the mean deviations of x
  x_list_bar = [i - x_bar for i in x_list]
  # Calculating the denominator of slope
  slope_denom = sum([i**2 for i in x_list_bar])

  # Convert list mean deviations of x to a Spark DataFrame column
  x_list_bar_col = F.array([F.lit(x) for x in x_list_bar])
  # x_list_col = F.array([F.lit(x) for x in x_list])

  
  w = Window.partitionBy(key_var).orderBy(time_var).rowsBetween(-n+1, 0)
  # Converting the denominator and mean deviation of x to spark dataframe columns
  pdf = pdf.withColumn('denom', F.lit(slope_denom))
  pdf = pdf.withColumn('x_bar_diff', F.lit(x_list_bar_col))
  # Generating a column containing arrays of the previous n values
  pdf = pdf.withColumn('y_col', F.collect_list(var).over(w))

  # Calculating the rolling mean over the period n

  pdf = pdf.withColumn('y_bar', F.avg(var).over(w))

  # Generating the mean deviations of y
  pdf = pdf.withColumn('y_bar_diff', F.expr('transform(y_col, x -> x - y_bar)'))

  
  # Define a UDF to calculate the dot product
  def dot_product(v1, v2):
      return sum([x * y for x, y in zip(v1, v2)])
    
  dot_product_udf = F.udf(dot_product, DoubleType())

  # Calculate the dot product of mean deviations of y and x which sthe numerator of the slope coefficient
  pdf = pdf.withColumn("x_y_col", dot_product_udf(F.col("x_bar_diff"), F.col("y_bar_diff")))

  # Calculate the n period slope
  pdf = pdf.withColumn(var+'_trend_'+str(n), F.col('x_y_col')/F.col('denom'))

  return pdf.drop(*['denom','x_bar_diff','y_col','y_bar','y_bar_diff','x_y_col'])

def shape(pdf: pyspark.sql.dataframe.DataFrame, print_shape: bool = False):
  """
  Prints the shape of the pyspark sql dataframe (Number of rows and columns)

  Args:
    pdf (pyspark.sql.dataframe.DataFrame): The pyspark input dataframe
    print (bool): If True prints the shape, else returns a tuple. Default is True

  Returns:
    str: The shape of the dataframe
  """
  if print_shape:
    print(f"Rows:  {pdf.count()}")
    print(f"Columns:  {len(pdf.columns)}")
  else:
    return (pdf.count(), len(pdf.columns))


def rolling_master(pdf: pyspark.sql.dataframe.DataFrame, period_list: list, metrics_list: list, non_feature_list: list, key_var: str, time_var: str, all_period_trend: bool = True) -> pyspark.sql.dataframe.DataFrame:

  """
    Get the rolling aggregates and slope coefficient of a numeric column based on a key and time variable 

    For example, if the period list is [3,6] and the metric_list is [avg, max] then it will generate
    1. avg & max of last 3 periods
    2. avg & max of last 6 periods
    3. latest / avg of last 3 periods
    4. latest / avg of last 6 periods
    5. avg of last 3 periods / avg of last 6 periods
    6. avg of last 3 periods / avg of pre last 3 periods

    If 'avg' not passed in the metrics_list then it will skip generating features from 3 to 6

    Args:
        pdf (pyspark.sql.dataframe.DataFrame): Input pyspark sql dataframe.
        period_list (list): List containing two periods on which the rolling aggregate is calculated
        metrics_list (list): List of aggregate function
        non_feature_list (list): List of columns excluded from the rolling aggregate generation
        key_var (str): Key column
        time_var (str): Time column
        all_period_trend (bool): If True will calculate the slope trend of both the periods else will only calculate for the highest period. Default: True

    Returns:
        pyspark.sql.dataframe.DataFrame: The original DataFrame adding the output rolling aggregated columns

    Raises:
        ValueError: When the first element of the period_list is not greater than 1
        ValueError: When the first element of the period_list is not less than the second element
        ValueError: When the length of the period_list is not 2

  """

  if len(period_list) == 2:
    if period_list[0]<period_list[1]:
      if period_list[0]>1:
        feature_columns = [c for c in pdf.columns if c not in non_feature_list]

        for c in feature_columns:
          pdf = rolling_aggregate_pre(pdf, c, period_list[0], 'avg', key_var, time_var)
          for m in metrics_list:
            for p in period_list:
              pdf = rolling_aggregate(pdf, c, p, m, key_var, time_var)
              

        first_period_cols = [c for c in pdf.columns if c.endswith('_'+str(period_list[0])) & c.startswith('avg')]

        second_period_cols = [c for c in pdf.columns if c.endswith('_'+str(period_list[1])) & c.startswith('avg')]

        # latest vs avg of first period vars
        if len(feature_columns) == len(first_period_cols):
          for i,j in tuple(zip(feature_columns,first_period_cols)):
            pdf = get_ratio(pdf, i+'_1v'+str(period_list[0]), i, j, 0)
        else:
          print('Skipped latest vs avg of first period vars')

        # latest vs avg of second period vars
        if len(feature_columns) == len(second_period_cols):
          for i,j in tuple(zip(feature_columns,second_period_cols)):
            pdf = get_ratio(pdf, i+'_1v'+str(period_list[1]), i, j, 0)
        else:
          print('Skipped latest vs avg of second period vars')

        # avg of first period v avg of second period vars
        if len(feature_columns) == len(first_period_cols) == len(second_period_cols) :
          for i,j,k in tuple(zip(feature_columns,first_period_cols,second_period_cols)):
            pdf = get_ratio(pdf, i+'_'+str(period_list[0])+'v'+str(period_list[1]), j, k, 0)
        else:
          print('Skipped avg of first period v avg of second period vars')

        # avg of first period v avg of pre first period vars
        first_period_cols_pre = [c for c in pdf.columns if c.endswith('_'+str(period_list[0])+'_pre') & c.startswith('avg')]
        pdf = pdf.fillna(0, subset=first_period_cols_pre)

        if len(feature_columns) == len(first_period_cols) == len(first_period_cols_pre) :
          for i,j,k in tuple(zip(feature_columns,first_period_cols,first_period_cols_pre)):
            pdf = get_ratio(pdf, i+'_'+str(period_list[0])+'v'+str(period_list[0]), j, k, 0)
        else:
          print('Skipped avg of first period v avg of pre first period vars')

        pdf = pdf.drop(*first_period_cols_pre)

        if all_period_trend:
          for c in feature_columns:
            for p in period_list:
              pdf = trend_coeff(pdf, c, p, key_var, time_var)
        else:
          for c in feature_columns:
            pdf = trend_coeff(pdf, c, period_list[1], key_var, time_var)

        return pdf
 
      else:
        raise ValueError("the first element of period_list should be greater than one")
    else:
      raise ValueError("the first element of period_list should be less than the second element")
  else:
    raise ValueError("length of period_list should be 2")
