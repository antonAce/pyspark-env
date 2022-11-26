import re

from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, StringType


def sub_usertags(text: str, sub_token="<USER>") -> str:
    # Substitute usernames (starts with '@' symbol) with a custom token
    return re.sub(r'@[\S]+', sub_token, text)

def sub_hashtags(text: str, sub_token=None) -> str:
    # Substitute #hashtags with a custom token (if None, substitute with the inner word)
    return re.sub(r'(#)([\S]+)', sub_token if sub_token is not None else r'\2', text)

def sub_unicode(text: str, sub_token=' ') -> str:
    # Substitute Unicode symbols
    return re.sub(r'[^\u0000-\u007F]+', sub_token, text)

def sub_sepr(text: str, sub_token=' ') -> str:
    # Substitute common line separation symbols (\n, \r, \t)
    return re.sub(r'[\n\r\t]+', sub_token, text)

def sub_punc(text: str, sub_token=' ') -> str:
    # Substitute punctuation
    return re.sub(r'[!"#$%&\(\)\*\+,\./:;<=>?@\\^_`{|}~\[\]-]+', sub_token, text)

def sub_num_words(text: str, sub_token='') -> str:
    # Substitute words that contain digits
    return re.sub(r'\b(\w)*(\d)(\w)*\b', sub_token, text)

def sub_space(text: str, sub_token=' ') -> str:
    # Remove redundant spaces between each word
    return re.sub(r'[ ]+', sub_token, text)

def social_proc(text: str) -> str:
    # Preprocess text from a social media
    steps = [
        lambda text: text.lower(),
        lambda text: sub_usertags(text),
        lambda text: sub_hashtags(text),
        lambda text: sub_sepr(text),
        lambda text: sub_num_words(text),
        lambda text: sub_space(text),
        lambda text: text.strip()
    ]

    for step in steps:
        text = step(text)

    return text


# Preprocessing UDFs
strip_udf = udf(lambda text: text.strip(), StringType())
normalise_udf = udf(lambda text: text.lower(), StringType())
sub_usertags_udf = udf(lambda text: sub_usertags(text), StringType())
sub_hashtags_udf = udf(lambda text: sub_hashtags(text), StringType())
sub_unicode_udf = udf(lambda text: sub_unicode(text), StringType())
sub_sepr_udf = udf(lambda text: sub_sepr(text), StringType())
sub_punc_udf = udf(lambda text: sub_punc(text), StringType())
sub_num_words_udf = udf(lambda text: sub_num_words(text), StringType())
sub_space_udf = udf(lambda text: sub_space(text), StringType())

# Metrics UDFs
wordlen_udf = udf(lambda text: len(text.split(' ')), IntegerType())
charlen_udf = udf(lambda text: len(text), IntegerType())

# Custom UDFs
social_proc_udf = udf(lambda text: social_proc(text), StringType())
