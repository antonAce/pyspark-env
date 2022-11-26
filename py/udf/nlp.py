import re

from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, StringType


def sub_usertags(text: str, sub_token="<USER>") -> str:
    # Substitute usernames (starts with '@' symbol) with a custom token
    return re.sub(r'@[\S]+', sub_token, text)

def sub_hashtags(text: str, sub_token=None) -> str:
    # Substitute #hashtags with a custom token (if None, substitute with the inner word)
    return re.sub(r'(#)([\S]+)', sub_token if sub_token is not None else r'\2', text)


# Preprocessing UDFs
sub_usertags_udf = udf(lambda text: sub_usertags(text), StringType())
sub_hashtags_udf = udf(lambda text: sub_hashtags(text), StringType())

# Metrics UDFs
wordlen_udf = udf(lambda text: len(text.split(' ')), IntegerType())
charlen_udf = udf(lambda text: len(text), IntegerType())
