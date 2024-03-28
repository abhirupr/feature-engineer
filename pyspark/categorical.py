from databricks.sdk.runtime import *
# import pyspark.sql.SparkSession as spark
from sklearn.model_selection import KFold
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.window import Window

class KFoldTargetEncoderTrain:
  def __init__(self, cat_col, target_col,
                n_fold=5, verbosity=True,
                drop_target_col=False):
    self.cat_col = cat_col
    self.target_col = target_col
    self.n_fold = n_fold
    self.verbosity = verbosity
    self.drop_target_col = drop_target_col

  def transform(self, X):
    assert isinstance(self.target_col, str)
    assert isinstance(self.cat_col, str)
    assert self.cat_col in X.columns
    assert self.target_col in X.columns

    # Calculate mean of target
    mean_of_target = X.select(self.target_col).agg(F.mean(self.target_col)).collect()[0][0]

    # Column name for K-fold target encoding
    col_mean_name = f"{self.cat_col}_kfold_target_enc"

    # Add index col to dataframe
    windowSpec = Window.orderBy(F.monotonically_increasing_id())
    X = X.select("*").withColumn("id", F.row_number().over(windowSpec).cast('Integer')-1)

    # Define the schema of the output DataFrame
    fields = X.schema.fields + [StructField(col_mean_name, DoubleType(), nullable=True)]
    schema = StructType(fields)

    # Create an empty DataFrame with the defined output schema
    X_final = spark.createDataFrame([], schema)

    # Perform K-fold target encoding
    for tr_ind, val_ind in self._kfold_indices(X):
      X_tr, X_val = X.filter(F.col("id").isin(tr_ind)), X.filter(F.col("id").isin(val_ind))
      mean_encoded_values = X_tr.groupBy(self.cat_col).agg(F.mean(self.target_col).alias(col_mean_name))
      X_val = X_val.join(mean_encoded_values, on=self.cat_col, how="left").select(X_final.columns)
      X_val = X_val.withColumn(col_mean_name, F.when(F.col(col_mean_name).isNull(), mean_of_target).otherwise(F.col(col_mean_name)))
      X_final = X_final.union(X_val)

    # Print correlation if verbosity is enabled
    if self.verbosity:
      corr_value = X_final.corr(self.target_col, col_mean_name)
      print(f"Correlation between the new feature, {col_mean_name}, and {self.target_col} is {corr_value}.")

    # Drop original column if required
    if self.drop_target_col:
      X_final = X_final.drop(self.target_col)

    return X_final.sort('id').drop('id')

  def _kfold_indices(self, X):
    # Generate K-fold indices
    kf = KFold(n_splits=self.n_fold, shuffle=False, random_state=None)
    for tr_ind, val_ind in kf.split(X.toPandas()):
      yield tr_ind.tolist(), val_ind.tolist()

class KFoldTargetEncoderTest:
  def __init__(self, pdf, cat_col, cat_encode_col):
    self.pdf = pdf
    self.cat_col = cat_col
    self.cat_encode_col = cat_encode_col

  def transform(self, X):
    assert isinstance(self.cat_col, str)
    assert isinstance(self.cat_encode_col, str)
    assert self.cat_col in self.pdf.columns
    assert self.cat_col in X.columns
    assert self.cat_encode_col in self.pdf.columns
     

    # Calculate mean of the target column
    mean_encoded_values = self.pdf.groupBy(self.cat_col).agg(F.mean(self.cat_encode_col).alias(self.cat_encode_col))

    # Join mean_encoded_values with X on the specified column
    X = X.join(mean_encoded_values, on=self.cat_col, how="left")

    # Replace missing values with the overall mean
    X = X.withColumn(self.cat_encode_col
                     , F.when(F.col(self.cat_encode_col).isNull()
                              , mean_encoded_values.select(self.cat_encode_col).agg(F.mean(self.cat_encode_col)).collect()[0][0])\
                                .otherwise(F.col(self.cat_encode_col)))

    return X