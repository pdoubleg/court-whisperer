import logging
from typing import List, Dict, Optional, no_type_check
import numpy as np
import pandas as pd
from csv import Sniffer
import tiktoken
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from langchain.schema import Document

from src.parsing.utils import download_nltk_resource

logger = logging.getLogger(__name__)


def dataframe_to_lc_documents(df: pd.DataFrame, content: str, id_col: Optional[str] = None, metadata: List[str] = None) -> List[Document]:
    """
    Convert a DataFrame to a list of Document objects.

    Args:
        df (DataFrame): Input DataFrame.
        content_col (str): Column name in df to use as content for Document.
        id_col (str, optional): Column name in df to use as id for Document. Defaults to None.
        metadata (List[str], optional): List of column names in df to use as metadata for Document. Defaults to None.

    Returns:
        List[Document]: List of Document objects.
    """
    documents = []
    for i, row in df.iterrows():
        meta: Dict[str, str] = {key: row[key] for key in metadata} if metadata else {}
        meta["id_"] = row[id_col] if id_col else i
        doc = Document(
            page_content=row[content],
            metadata=meta
        )
        documents.append(doc)
    return documents


def lc_documents_to_dataframe(documents: List[Document], id_col: Optional[str] = None) -> pd.DataFrame:
    """
    Convert a list of Document objects to a DataFrame.

    Args:
        documents (List[Document]): List of Document objects.
        id_col (str, optional): Column name in df to use as id for Document. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame representation of the Document objects.
    """
    data = []
    for doc in documents:
        row = doc.metadata
        row['content'] = doc.page_content
        if id_col:
            row[id_col] = doc.metadata.get('id_')
        data.append(row)
    df = pd.DataFrame(data)
    return df


def preprocess_text(text: str) -> str:
    """
    Preprocesses the given text by:
    1. Lowercasing all words.
    2. Tokenizing (splitting the text into words).
    3. Removing punctuation.
    4. Removing stopwords.
    5. Lemmatizing words.

    Args:
        text (str): The input text.

    Returns:
        str: The preprocessed text.
    """
    # Ensure the NLTK resources are available
    for resource in ["punkt", "wordnet", "stopwords"]:
        download_nltk_resource(resource)

    # Lowercase the text
    text = text.lower()

    # Tokenize the text and remove punctuation
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    # Join the words back into a string
    text = " ".join(tokens)

    return text


def preprocess_doc_text(docs):
    return [
            Document(content=preprocess_text(d.page_content), metadata=d.metadata)
            for d in docs
        ]


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def read_tabular_data(path_or_url: str, sep: None | str = None) -> pd.DataFrame:
    """
    Reads tabular data from a file or URL and returns a pandas DataFrame.
    The separator is auto-detected if not specified.

    Args:
        path_or_url (str): Path or URL to the file to be read.

    Returns:
        pd.DataFrame: Data from file or URL as a pandas DataFrame.

    Raises:
        ValueError: If the data cannot be read or is misformatted.
    """
    try:
        if sep is None:
            # Read the first few lines to guess the separator
            with pd.io.common.get_handle(path_or_url, "r") as file_handler:
                first_lines = "".join(file_handler.handle.readlines(5))
                sep = Sniffer().sniff(first_lines).delimiter
                # If it's a local file, reset to the beginning
                if hasattr(file_handler.handle, "seek"):
                    file_handler.handle.seek(0)

        # Read the data

        # get non-blank column names
        with pd.io.common.get_handle(path_or_url, "r") as f:
            header_line = f.handle.readline().strip()
            valid_cols = [col for col in header_line.split(sep) if col]
            valid_cols = [c.replace('"', "").replace("'", "") for c in valid_cols]
            if hasattr(f.handle, "seek"):
                f.handle.seek(0)

        # use only those columns
        data = pd.read_csv(path_or_url, sep=sep, usecols=valid_cols)
        data.columns = data.columns.str.strip()  # e.g. "  column 1  " -> "column 1"

        return data

    except Exception as e:
        raise ValueError(
            "Unable to read data. "
            "Please ensure it is correctly formatted. Error: " + str(e)
        )


def describe_dataframe(df: pd.DataFrame, sample_size: int = 5) -> str:
    """
    Generates a description of the columns in the dataframe, along with typical values.
    Intended to be used to insert into an LLM context so it can generate
    appropriate queries or filters on the df.

    Args:
        df (pd.DataFrame): The dataframe to describe.
        sample_size (int): The number of sample values to show for each column.

    Returns:
        str: A description of the dataframe.
    """
    description = []
    for column in df.columns:
        sample_values = df[column].dropna().head(sample_size).tolist()
        if len(sample_values) > 0 and isinstance(sample_values[0], str):
            # truncate to 100 chars
            sample_values = [v[:100] for v in sample_values]
        col_type = "string" if df[column].dtype == "object" else df[column].dtype
        col_desc = f"* {column} ({col_type}): {sample_values}"
        description.append(col_desc)

    all_cols = "\n".join(description)

    return f"""
        Name of each field, its type and some typical values:
        {all_cols}
        """


@no_type_check
def dataframe_summary(df: pd.DataFrame) -> str:
    """
    Generate a structured summary for a pandas DataFrame containing numerical
    and categorical values.

    Args:
        df (pd.DataFrame): The input DataFrame to summarize.

    Returns:
        str: A nicely structured and formatted summary string (to send to LLM).
    """

    # Column names display
    col_names_str = (
        "COLUMN NAMES:\n" + " ".join([f"'{col}'" for col in df.columns]) + "\n\n"
    )

    # Numerical data summary
    num_summary = df.describe().map(lambda x: "{:.2f}".format(x))
    num_str = "Numerical Column Summary:\n" + num_summary.to_string() + "\n\n"

    # Categorical data summary
    cat_columns = df.select_dtypes(include=[np.object_]).columns
    cat_summary_list = []

    for col in cat_columns:
        unique_values = df[col].unique()
        if len(unique_values) < 10:
            cat_summary_list.append(f"'{col}': {', '.join(map(str, unique_values))}")
        else:
            cat_summary_list.append(f"'{col}': {df[col].nunique()} unique values")

    cat_str = "Categorical Column Summary:\n" + "\n".join(cat_summary_list) + "\n\n"

    # Missing values summary
    nan_summary = df.isnull().sum().rename("missing_values").to_frame()
    nan_str = "Missing Values Column Summary:\n" + nan_summary.to_string() + "\n"

    # Combine the summaries into one structured string
    summary_str = col_names_str + num_str + cat_str + nan_str

    return summary_str

