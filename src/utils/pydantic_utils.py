import logging
import numpy as np
import pandas as pd
import json
from docstring_parser import parse
from functools import wraps
from pydantic import BaseModel, create_model
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    get_args,
    get_origin,
)

from src.types import Document, DocMetaData

logger = logging.getLogger(__name__)


def has_field(model_class: Type[BaseModel], field_name: str) -> bool:
    """Check if a Pydantic model class has a field with the given name."""
    return field_name in model_class.__fields__

def _recursive_purge_dict_key(d: Dict[str, Any], k: str) -> None:
    """Remove a key from a dictionary recursively"""
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == k and "type" in d.keys():
                del d[key]
            else:
                _recursive_purge_dict_key(d[key], k)
                
def flatten_pydantic_model(
    model: Type[BaseModel],
    base_model: Type[BaseModel] = BaseModel,
) -> Type[BaseModel]:
    """
    Given a possibly nested Pydantic class, return a flattened version of it,
    by constructing top-level fields, whose names are formed from the path
    through the nested structure, separated by double underscores.

    Args:
        model (Type[BaseModel]): The Pydantic model to flatten.
        base_model (Type[BaseModel], optional): The base model to use for the
            flattened model. Defaults to BaseModel.

    Returns:
        Type[BaseModel]: The flattened Pydantic model.
    """

    flattened_fields: Dict[str, Any] = {}
    models_to_process = [(model, "")]

    while models_to_process:
        current_model, current_prefix = models_to_process.pop()

        for name, field in current_model.__fields__.items():
            if isinstance(field.outer_type_, type) and issubclass(
                field.outer_type_, BaseModel
            ):
                new_prefix = (
                    f"{current_prefix}{name}__" if current_prefix else f"{name}__"
                )
                models_to_process.append((field.outer_type_, new_prefix))
            else:
                flattened_name = f"{current_prefix}{name}"

                if field.default_factory is not field.default_factory:
                    flattened_fields[flattened_name] = (
                        field.outer_type_,
                        field.default_factory,
                    )
                elif field.default is not field.default:
                    flattened_fields[flattened_name] = (
                        field.outer_type_,
                        field.default,
                    )
                else:
                    flattened_fields[flattened_name] = (field.outer_type_, ...)

    return create_model("FlatModel", __base__=base_model, **flattened_fields)


def flatten_pydantic_instance(
    instance: BaseModel,
    prefix: str = "",
    force_str: bool = False,
) -> Dict[str, Any]:
    """
    Given a possibly nested Pydantic instance, return a flattened version of it,
    as a dict where nested traversal paths are translated to keys a__b__c.

    Args:
        instance (BaseModel): The Pydantic instance to flatten.
        prefix (str, optional): The prefix to use for the top-level fields.
        force_str (bool, optional): Whether to force all values to be strings.

    Returns:
        Dict[str, Any]: The flattened dict.

    """
    flat_data: Dict[str, Any] = {}
    for name, value in instance.dict().items():
        # Assuming nested pydantic model will be a dict here
        if isinstance(value, dict):
            nested_flat_data = flatten_pydantic_instance(
                instance.__fields__[name].type_(**value),
                prefix=f"{prefix}{name}__",
                force_str=force_str,
            )
            flat_data.update(nested_flat_data)
        else:
            flat_data[f"{prefix}{name}"] = str(value) if force_str else value
    return flat_data


def nested_dict_from_flat(
    flat_data: Dict[str, Any],
    sub_dict: str = "",
) -> Dict[str, Any]:
    """
    Given a flattened version of a nested dict, reconstruct the nested dict.
    Field names in the flattened dict are assumed to be of the form
    "field1__field2__field3", going from top level down.

    Args:
        flat_data (Dict[str, Any]): The flattened dict.
        sub_dict (str, optional): The name of the sub-dict to extract from the
            flattened dict. Defaults to "" (extract the whole dict).

    Returns:
        Dict[str, Any]: The nested dict.

    """
    nested_data: Dict[str, Any] = {}
    for key, value in flat_data.items():
        if sub_dict != "" and not key.startswith(sub_dict + "__"):
            continue
        keys = key.split("__")
        d = nested_data
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    if sub_dict != "":  # e.g. "payload"
        nested_data = nested_data[sub_dict]
    return nested_data


def pydantic_obj_from_flat_dict(
    flat_data: Dict[str, Any],
    model: Type[BaseModel],
    sub_dict: str = "",
) -> BaseModel:
    """Flattened dict with a__b__c style keys -> nested dict -> pydantic object"""
    nested_data = nested_dict_from_flat(flat_data, sub_dict)
    return model(**nested_data)


def clean_schema(model: Type[BaseModel], excludes: List[str] = []) -> Dict[str, Any]:
    """
    Generate a simple schema for a given Pydantic model,
    including inherited fields, with an option to exclude certain fields.
    Handles cases where fields are Lists or other generic types and includes
    field descriptions if available.

    Args:
        model (Type[BaseModel]): The Pydantic model class.
        excludes (List[str]): A list of field names to exclude.

    Returns:
        Dict[str, Any]: A dictionary representing the simple schema.
    """
    schema = {}

    for field_name, field_info in model.__fields__.items():
        if field_name in excludes:
            continue

        field_type = field_info.outer_type_
        description = field_info.field_info.description or ""

        # Handle generic types like List[...]
        if get_origin(field_type):
            inner_types = get_args(field_type)
            inner_type_names = [
                t.__name__ if hasattr(t, "__name__") else str(t) for t in inner_types
            ]
            field_type_str = (
                f"{get_origin(field_type).__name__}" f'[{", ".join(inner_type_names)}]'
            )
            schema[field_name] = {"type": field_type_str, "description": description}
        elif issubclass(field_type, BaseModel):
            # Directly use the nested model's schema,
            # integrating it into the current level
            nested_schema = clean_schema(field_type, excludes)
            schema[field_name] = {**nested_schema, "description": description}
        else:
            # For basic types, use 'type'
            schema[field_name] = {
                "type": field_type.__name__,
                "description": description,
            }

    return schema


def numpy_to_python_type(numpy_type: Type[Any]) -> Type[Any]:
    """Converts a numpy data type to its Python equivalent."""
    type_mapping = {
        np.float64: float,
        np.float32: float,
        np.int64: int,
        np.int32: int,
        np.bool_: bool,
        # Add other numpy types as necessary
    }
    return type_mapping.get(numpy_type, numpy_type)


def dataframe_to_pydantic_model(df: pd.DataFrame) -> Type[BaseModel]:
    """Make a Pydantic model from a dataframe."""
    fields = {col: (type(df[col].iloc[0]), ...) for col in df.columns}
    return create_model("DataFrameModel", __base__=BaseModel, **fields)  # type: ignore


def dataframe_to_pydantic_objects(df: pd.DataFrame) -> List[BaseModel]:
    """Make a list of Pydantic objects from a dataframe."""
    Model = dataframe_to_pydantic_model(df)
    return [Model(**row.to_dict()) for index, row in df.iterrows()]


def first_non_null(series: pd.Series) -> Any | None:
    """Find the first non-null item in a pandas Series."""
    for item in series:
        if item is not None:
            return item
    return None


def dataframe_to_document_model(
    df: pd.DataFrame,
    content: str = "content",
    metadata: List[str] = [],
    exclude: List[str] = [],
) -> Type[BaseModel]:
    """
    Make a subclass of Document from a dataframe.

    Args:
        df (pd.DataFrame): The dataframe.
        content (str): The name of the column containing the content,
            which will map to the Document.content field.
        metadata (List[str]): A list of column names containing metadata;
            these will be included in the Document.metadata field.
        exclude (List[str]): A list of column names to exclude from the model.

    Returns:
        Type[BaseModel]: A pydantic model subclassing Document.
    """

    # Remove excluded columns
    df = df.drop(columns=exclude, inplace=False)
    # Check if metadata_cols is empty

    if metadata:
        # Define fields for the dynamic subclass of DocMetaData
        metadata_fields = {
            col: (
                Optional[numpy_to_python_type(type(first_non_null(df[col])))],
                None,  # Optional[numpy_to_python_type(type(first_non_null(df[col])))],
            )
            for col in metadata
        }
        DynamicMetaData = create_model(  # type: ignore
            "DynamicMetaData", __base__=DocMetaData, **metadata_fields
        )
    else:
        # Use the base DocMetaData class directly
        DynamicMetaData = DocMetaData

    # Define additional top-level fields for DynamicDocument
    additional_fields = {
        col: (
            Optional[numpy_to_python_type(type(first_non_null(df[col])))],
            None,  # Optional[numpy_to_python_type(type(first_non_null(df[col])))],
        )
        for col in df.columns
        if col not in metadata and col != content
    }

    # Create a dynamic subclass of Document
    DynamicDocumentFields = {
        **{"metadata": (DynamicMetaData, ...)},
        **additional_fields,
    }
    DynamicDocument = create_model(  # type: ignore
        "DynamicDocument", __base__=Document, **DynamicDocumentFields
    )

    def from_df_row(
        cls: type[BaseModel],
        row: pd.Series,
        content: str = "content",
        metadata: List[str] = [],
    ) -> BaseModel | None:
        content_val = row[content] if (content and content in row) else ""
        metadata_values = (
            {col: row[col] for col in metadata if col in row} if metadata else {}
        )
        additional_values = {
            col: row[col] for col in additional_fields if col in row and col != content
        }
        metadata = DynamicMetaData(**metadata_values)
        return cls(content=content_val, metadata=metadata, **additional_values)

    # Bind the method to the class
    DynamicDocument.from_df_row = classmethod(from_df_row)

    return DynamicDocument  # type: ignore


def dataframe_to_documents(
    df: pd.DataFrame,
    content: str = "content",
    metadata: List[str] = [],
    doc_cls: Type[BaseModel] | None = None,
) -> List[Document]:
    """
    Make a list of Document objects from a dataframe.
    Args:
        df (pd.DataFrame): The dataframe.
        content (str): The name of the column containing the content,
            which will map to the Document.content field.
        metadata (List[str]): A list of column names containing metadata;
            these will be included in the Document.metadata field.
        doc_cls (Type[BaseModel], optional): A Pydantic model subclassing
            Document. Defaults to None.
    Returns:
        List[Document]: The list of Document objects.
    """
    Model = doc_cls or dataframe_to_document_model(df, content, metadata)
    docs = [
        Model.from_df_row(row, content, metadata)  # type: ignore
        for _, row in df.iterrows()
    ]
    return [m for m in docs if m is not None]


class OpenAISchema(BaseModel):
    """
    Augments a Pydantic model with OpenAI's schema for function calling

    This class augments a Pydantic model with OpenAI's schema for function calling. The schema is generated from the model's signature and docstring. The schema can be used to validate the response from OpenAI's API and extract the function call.

    ## Usage

    ```python

    class User(OpenAISchema):
        first_name: str
        age: int

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "content": "Jason is 20 years old",
            "role": "user"
        }],
        functions=[User.openai_schema],
        function_call={"name": User.openai_schema["name"]},
    )

    user = User.from_response(completion)

    print(user)
    ```
    ## Result

    ```
    User(first_name='Jason', age=20)
        
    ```


    """

    @classmethod
    @property
    def openai_schema(cls):
        """
        Return the schema in the format of OpenAI's schema as jsonschema

        Note:
            Its important to add a docstring to describe how to best use this class, it will be included in the description attribute and be part of the prompt.

        Returns:
            model_json_schema (dict): A dictionary in the format of OpenAI's schema as jsonschema
        """
        schema = cls.schema()
        docstring = parse(cls.__doc__ or "")
        parameters = {
            k: v for k, v in schema.items() if k not in ("title", "description")
        }
        for param in docstring.params:
            if (name := param.arg_name) in parameters["properties"] and (
                description := param.description
            ):
                if "description" not in parameters["properties"][name]:
                    parameters["properties"][name]["description"] = description

        parameters["required"] = sorted(
            k for k, v in parameters["properties"].items() if "default" not in v
        )

        if "description" not in schema:
            if docstring.short_description:
                schema["description"] = docstring.short_description
            else:
                schema["description"] = (
                    f"Correctly extracted `{cls.__name__}` with all "
                    f"the required parameters with correct types"
                )

        return {
            "name": schema["title"],
            "description": schema["description"],
            "parameters": parameters,
        }

    @classmethod
    def from_response(cls, completion, throw_error=True):
        """Execute the function from the response of an openai chat completion

        Parameters:
            completion (openai.ChatCompletion): The response from an openai chat completion
            throw_error (bool): Whether to throw an error if the function call is not detected

        Returns:
            cls (OpenAISchema): An instance of the class
        """
        message = completion.choices[0].message

        if throw_error:
            assert "function_call" in message, "No function call detected"
            assert (
                message["function_call"]["name"] == cls.openai_schema["name"]
            ), "Function name does not match"

        function_call = message["function_call"]
        arguments = json.loads(function_call["arguments"], strict=False)
        return cls(**arguments)


def openai_schema(cls) -> OpenAISchema:
    if not issubclass(cls, BaseModel):
        raise TypeError("Class must be a subclass of pydantic.BaseModel")

    return wraps(cls, updated=())(
        create_model(
            cls.__name__,
            __base__=(cls, OpenAISchema),
        )
    )  # type: ignore