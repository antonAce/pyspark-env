from pyspark.sql import DataFrame
from pyspark.sql.functions import rand, array
from .udf import cols_to_arr_udf, cols_to_dict_udf


def shuffle_df(df: DataFrame, seed=None) -> DataFrame:
    """Return a DataFrame with randomly shuffled rows."""
    return df.orderBy(rand(seed=seed))


def clean_df(df: DataFrame, columns: list) -> DataFrame:
    """Basic cleaning operation (drop NULLs and duplicates) on selected columns."""
    return df.dropDuplicates(columns).dropna(subset=tuple(columns))


def glue_cols(df: DataFrame, cols: list, output_col: str) -> DataFrame:
    """Joins dataframe columns `cols` into an array-column as `output_col`."""
    return df.select(cols_to_arr_udf(array(*cols)).alias(output_col))


def map_cols(df: DataFrame, key_col: str, val_col: str, output_col: str) -> DataFrame:
    """Joins dataframe columns into a map-column `output_col`."""
    return df.select(cols_to_dict_udf(array(key_col, val_col)).alias(output_col))
