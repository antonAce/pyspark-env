from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, MapType, StringType


cols_to_arr_udf = udf(lambda arr: [str(x) for x in arr if x is not None], ArrayType(StringType()))
cols_to_dict_udf = udf(lambda kv: {str(k): str(v) for k, v in zip(*kv)}, MapType(StringType(), StringType()))
