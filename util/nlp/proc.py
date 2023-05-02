from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import monotonically_increasing_id, lag


def dialog_lag(
    df: DataFrame, input_col: str, output_col: str, partition_col: str = None
) -> DataFrame:
    """Offset a text column in dataframe up by one to create (dialog-response) pairs from a text data.

    Args:
        df (DataFrame): A Pyspark dataframe to process.
        input_col (str): Name of existing column to process.
        output_col (str): Name of a new column to store column lag.
        partition_col (str, optional): Pass a partition column for a distributed processing. Defaults to None.

    Returns:
        DataFrame: A new dataframe with (dialog-response) pairs.
    """

    df = df.withColumn("id", monotonically_increasing_id())
    window = (
        Window.partitionBy(partition_col).orderBy("id")
        if partition_col is not None
        else Window.orderBy("id")
    )
    return df.withColumn(output_col, lag(input_col, -1).over(window))
