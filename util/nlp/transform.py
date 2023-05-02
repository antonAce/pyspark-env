from pyspark import keyword_only

from pyspark.ml import Transformer
from pyspark.ml.param.shared import (
    HasInputCols,
    HasOutputCol,
    Param,
    Params,
    TypeConverters,
)
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

from pyspark.sql import DataFrame

from .udf import cosine_sim_udf


class CosineSimTransformer(
    Transformer,
    HasInputCols,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    input_cols = Param(
        Params._dummy(),
        "input_cols",
        "A pair of input one-dimensional vectors.",
        typeConverter=TypeConverters.toListString,
    )
    result_col = Param(
        Params._dummy(),
        "result_col",
        "Cosine similarity vector.",
        typeConverter=TypeConverters.toString,
    )

    @keyword_only
    def setParams(self, input_cols=None, result_col=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def HasInputCols(self):
        return self.getOrDefault(self.input_cols)

    def HasOutputCol(self):
        return self.getOrDefault(self.result_col)

    def setInputCols(self, new_input_cols):
        return self.setParams(input_cols=new_input_cols)

    def setOutputCol(self, new_result_col):
        return self.setParams(result_col=new_result_col)

    def _transform(self, df: DataFrame):
        return df.withColumn(self.getOutputCol(), cosine_sim_udf(self.getInputCols()))
