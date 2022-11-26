from pyspark.sql import DataFrame
from pyspark.sql.functions import rand


def shuffle_df(df: DataFrame, seed=None) -> DataFrame:
    # Return a DataFrame with randomly shuffled rows
    return df.orderBy(rand(seed=seed))
