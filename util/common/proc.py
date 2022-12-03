from pyspark.sql import DataFrame
from pyspark.sql.functions import rand


def shuffle_df(df: DataFrame, seed=None) -> DataFrame:
    # Return a DataFrame with randomly shuffled rows
    return df.orderBy(rand(seed=seed))

def clean_df(df: DataFrame, columns: list) -> DataFrame:
    # Basic cleaning operation (drop NULLs and duplicates) on selected columns
    return df.dropDuplicates(columns).dropna(subset=tuple(columns))
