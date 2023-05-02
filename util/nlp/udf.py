import re
import numpy as np
import numpy.linalg as la

from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, StringType, FloatType

from .stopwords import STOP_WORDS


def sub_usertags(text: str, sub_token="<USER>") -> str:
    """Substitute usernames (starts with '@' symbol) with a custom token.

    Args:
        text (str): Input text string to process.
        sub_token (str, optional): A custom token to substitute for a username. Defaults to "<USER>".

    Returns:
        str: Processed input string.
    """

    return re.sub(r"@[\S]+", sub_token, text)


def sub_hashtags(text: str, sub_token=None) -> str:
    """Substitute hashtags (starts with '#' symbol) with a custom token.

    Args:
        text (str): Input text string to process.
        sub_token (str, optional): A custom token to substitute for a hashtag. If None, then function substitutes hashtag with the corresponding word. Defaults to None.

    Returns:
        str: Processed input string.
    """

    return re.sub(r"(#)([\S]+)", sub_token if sub_token is not None else r"\2", text)


def sub_unicode(text: str, sub_token=" ") -> str:
    """Substitute Unicode symbols.

    Args:
        text (str): Input text string to process.
        sub_token (str, optional): A custom token to substitute. Defaults to ' '.

    Returns:
        str: Processed input string.
    """

    return re.sub(r"[^\u0000-\u007F]+", sub_token, text)


def sub_sepr(text: str, sub_token=" ") -> str:
    """Substitute common line separation symbols (`\\n`, `\\r`, `\\t`).

    Args:
        text (str): Input text string to process.
        sub_token (str, optional): A custom token to substitute. Defaults to ' '.

    Returns:
        str: Processed input string.
    """

    return re.sub(r"[\n\r\t]+", sub_token, text)


def sub_punc(text: str, sub_token=" ") -> str:
    """Substitute punctuation symbols:
    ```['!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '.', '/', ':', ';', '<', '=', '>', '?', '@', '\\', '^', '_', '`', '{', '|', '}', '~', '[', ']', '-']```

    Args:
        text (str): Input text string to process.
        sub_token (str, optional): A custom token to substitute. Defaults to ' '.

    Returns:
        str: Processed input string.
    """

    return re.sub(r'[!"#$%&\(\)\*\+,\./:;<=>?@\\^_`{|}~\[\]-]+', sub_token, text)


def sub_url(text: str, sub_token="<URL>") -> str:
    """Substitute web resource link (URL) with a custom token.

    Args:
        text (str): Input text string to process.
        sub_token (str, optional): A custom token to substitute. Defaults to '<URL>'.

    Returns:
        str: Processed input string.
    """

    return re.sub(
        r"^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$",
        sub_token,
        text,
    )


def sub_numwords(text: str, sub_token="") -> str:
    """Substitute words that contain digits.

    Args:
        text (str): Input text string to process.
        sub_token (str, optional): A custom token to substitute. Defaults to ''.

    Returns:
        str: Processed input string.
    """

    return re.sub(r"\b(\w)*(\d)(\w)*\b", sub_token, text)


def sub_stopwords(text: str, sub_token="") -> str:
    """Substitute English stopwords.

    Args:
        text (str): Input text string to process.
        sub_token (str, optional): A custom token to substitute. Defaults to ''.

    Returns:
        str: Processed input string.
    """

    return " ".join(
        [word if word not in STOP_WORDS else sub_token for word in text.split(" ")]
    )


def sub_space(text: str, sub_token=" ") -> str:
    """Substitute 2+ adjacent space characters with a single space character.

    Args:
        text (str): Input text string to process.
        sub_token (str, optional): An alternative to a single space character. Defaults to ' '.

    Returns:
        str: Processed input string.
    """

    return re.sub(r"[ ]+", sub_token, text)


def iter_proc(text: str, steps=[]) -> str:
    """Iterative preprocessing of a text data.

    Args:
        text (str): Input text string to process.
        steps (list, optional): User defined processing steps of a `Callable` type. Defaults to [].

    Returns:
        str: Processed input string.
    """

    for step in steps:
        text = step(text)

    return text


def social_proc(text: str) -> str:
    """Preprocess text string related to a social media (e.g. a blog post): normalize and remove URLs, usernames, hashtags, separation symbols, words with numbers, and redundant spaces.

    Args:
        text (str): Input text string to process.

    Returns:
        str: Processed input string.
    """

    return iter_proc(
        text,
        steps=[
            lambda text: text.lower(),
            lambda text: sub_url(text),
            lambda text: sub_usertags(text),
            lambda text: sub_hashtags(text),
            lambda text: sub_sepr(text),
            lambda text: sub_numwords(text),
            lambda text: sub_space(text),
            lambda text: text.strip(),
        ],
    )


def full_proc(text: str) -> str:
    """Preprocess text with all available pipelines.

    Args:
        text (str): Input text string to process.

    Returns:
        str: Processed input string.
    """

    return iter_proc(
        text,
        steps=[
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
            lambda text: text.strip(),
        ],
    )


def cosine_sim(x: np.array, y: np.array) -> np.array:
    """Compute a cosine similarity between `x` and `y` vectors: `cos(x, y) = x @ y / (||x|| * ||y||)`

    Example:
        >>> cosine_sim(np.array([1.0, 1.0, 1.0, 1.0]), np.array([0.5, -3.0, 0.25, -1.0]))
        -0.5060243137049899

    Args:
        x (np.array): An input vector.
        y (np.array): Another input vector.

    Returns:
        np.array: Cosine similarity vector.
    """

    return x @ y / (la.norm(x) * la.norm(y))


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
wordlen_udf = udf(lambda text: len(text.split(" ")), IntegerType())
charlen_udf = udf(lambda text: len(text), IntegerType())

# Custom UDFs
social_proc_udf = udf(lambda text: social_proc(text), StringType())
full_proc_udf = udf(lambda text: full_proc(text), StringType())
