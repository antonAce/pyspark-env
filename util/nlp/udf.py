import re

from math import sqrt
from typing import List
from .stopwords import STOP_WORDS

from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, StringType, FloatType


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

def sub_url(text: str, sub_token='<URL>') -> str:
    # Substitute web resource link (URL) with a custom token
    return re.sub(r'^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$', sub_token, text)

def sub_numwords(text: str, sub_token='') -> str:
    # Substitute words that contain digits
    return re.sub(r'\b(\w)*(\d)(\w)*\b', sub_token, text)

def sub_stopwords(text: str, sub_token='') -> str:
    # Substitute english stopwords with a custom token 
    return ' '.join([word if word not in STOP_WORDS else sub_token for word in text.split(" ")])

def sub_space(text: str, sub_token=' ') -> str:
    # Remove redundant spaces between each word
    return re.sub(r'[ ]+', sub_token, text)

def iter_proc(text: str, steps=[]) -> str:
    # Iterative preprocessing of a text data
    for step in steps:
        text = step(text)

    return text

def social_proc(text: str) -> str:
    # Preprocess text from a social media
    return iter_proc(text, steps=[
        lambda text: text.lower(),
        lambda text: sub_url(text),
        lambda text: sub_usertags(text),
        lambda text: sub_hashtags(text),
        lambda text: sub_sepr(text),
        lambda text: sub_numwords(text),
        lambda text: sub_space(text),
        lambda text: text.strip()
    ])

def full_proc(text: str) -> str:
    # Preprocess text with all available pipelines
    return iter_proc(text, steps=[
        lambda text: text.lower(),
        lambda text: sub_url(text),
        lambda text: sub_usertags(text),
        lambda text: sub_hashtags(text),
        lambda text: sub_unicode(text),
        lambda text: sub_sepr(text),
        lambda text: sub_punc(text),
        lambda text: sub_numwords(text),
        lambda text: sub_stopwords(text),
        lambda text: sub_space(text),
        lambda text: text.strip()
    ])

def cosine_sim(x: List[float], y: List[float]) -> List[float]:
    """Compute a cosine similarity between `x` and `y` vectors: `cos(x, y) = x @ y / (||x|| * ||y||)`

    Args:
        x (List[float]): An input one-dimensional vector.
        y (List[float]): Another input one-dimensional vector.

    Returns:
        List[float]: Cosine similarity vector.
    """

    return (sum([xi * yi for xi, yi in zip(x, y)])) / (
        sqrt(sum([xi ** 2 for xi in x])) *
        sqrt(sum([yi ** 2 for yi in y]))
    )

# Preprocessing UDFs
strip_udf = udf(lambda text: text.strip(), StringType())
normalise_udf = udf(lambda text: text.lower(), StringType())
sub_url_udf = udf(lambda text: sub_url(text), StringType())
sub_usertags_udf = udf(lambda text: sub_usertags(text), StringType())
sub_hashtags_udf = udf(lambda text: sub_hashtags(text), StringType())
sub_unicode_udf = udf(lambda text: sub_unicode(text), StringType())
sub_sepr_udf = udf(lambda text: sub_sepr(text), StringType())
sub_punc_udf = udf(lambda text: sub_punc(text), StringType())
sub_numwords_udf = udf(lambda text: sub_numwords(text), StringType())
sub_stopwords_udf = udf(lambda text: sub_stopwords(text), StringType())
sub_space_udf = udf(lambda text: sub_space(text), StringType())

# Metrics UDFs
cosine_sim_udf = udf(lambda x, y: cosine_sim(x, y), FloatType())
wordlen_udf = udf(lambda text: len(text.split(' ')), IntegerType())
charlen_udf = udf(lambda text: len(text), IntegerType())

# Custom UDFs
social_proc_udf = udf(lambda text: social_proc(text), StringType())
full_proc_udf = udf(lambda text: full_proc(text), StringType())
