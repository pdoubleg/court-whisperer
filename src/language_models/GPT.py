import os
import logging
from typing import Any, Iterable, List, Optional, Type, Union, Dict, get_args, get_origin
from langchain.callbacks.manager import Callbacks
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage, ChatMessage, FunctionMessage
from pydantic import BaseModel, Field, root_validator, ValidationError
from json import JSONDecodeError

from src.utils.pydantic_utils import OpenAISchema, openai_schema

logger = logging.getLogger(__name__)

class GPT(ChatOpenAI):
    """
    GPT class that extends AzureChatOpenAI to allow it to function as an LLM and accept strings as inputs.
    """

    def __call__(
        self,
        messages: Union[List[BaseMessage], str],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[BaseMessage, str]:
        """
        Call method for the GPT class.

        Args:
            messages: A list of BaseMessage instances or a string.
            stop: An optional list of strings to stop the generation.
            callbacks: An optional Callbacks instance.
            **kwargs: Additional keyword arguments.

        Returns:
            A BaseMessage instance or a string depending on the input type.
        """
        return_str = False

        if isinstance(messages, str):
            return_str = True
            messages = [HumanMessage(content=messages)]

        result = super().__call__(messages, stop, callbacks, **kwargs)

        if return_str:
            return result.content
        else:
            return result

    @root_validator(pre=True)
    def set_api_key(cls, values):
        """
        Root validator to set the API key.

        Args:
            values: A dictionary of values.

        Returns:
            The updated dictionary of values.
        """
        if values.get("openai_api_key") is None:
            values["openai_api_key"] = values.get("cortex_api_token", os.environ.get("OPENAI_API_KEY"))

        return values

    @property
    def _client_params(self) -> Dict[str, Any]:
        """
        Property to get the client parameters.

        Returns:
            A dictionary of client parameters.
        """
        params = super()._client_params

        if "max_tokens" in params and params["max_tokens"] is None:
            # the GPT API coerces None to 0 causing errors
            del params["max_tokens"]

        return params
    
    
    def _handle_response_model(
        self,
        response_model: Type[BaseModel],
    ):  # type: ignore
        """
        Handles the response model conversion to OpenAISchema.

        Args:
            response_model: The response model to handle.

        Returns:
            The handled response model.
            
        """
        if response_model is not None:
            if not issubclass(response_model, OpenAISchema):
                response_model = openai_schema(response_model)  # type: ignore

        return response_model
    
    
    def _process_response(
        self,
        response: AIMessage,
        response_model: OpenAISchema,
        strict: Optional[bool] = False,
    ) -> Union[AIMessage, OpenAISchema]:  # type: ignore
        """
        Processes the response from openai.

        Args:
            response: The response to process.
            response_model: The response model to use for processing.
            strict: Whether to use strict processing.

        Returns:
            The processed response.
        """
        if response_model is not None:
            model = response_model.from_lc_response(
                response,
                strict=strict,
            )
            return model
        return response
    
    
    def _retry_sync(
        self,
        messages: List[BaseMessage],
        response_model: OpenAISchema,
        max_retries: int,
        strict: Optional[bool] = False,
    ): # type: ignore
        """
        Retries validation errors by passing them back to the llm.

        Args:
            messages: The messages to process.
            response_model: The response model to use for syncing.
            max_retries: The maximum number of retries.
            strict: Whether to use strict syncing.

        Returns:
            The result of the sync operation.
        """
        retries = 0
        while retries <= max_retries:
            # Excepts ValidationError, and JSONDecodeError
            try:
                response = self.predict_messages(
                    messages, 
                    functions=[response_model.openai_schema],
                    function_call={"name": response_model.openai_schema["name"]},
                    )

                return self._process_response(
                    response,
                    response_model=response_model,
                    strict=strict,
                )
            except (ValidationError, JSONDecodeError) as e:
                
                bad_response = FunctionMessage(
                    content=str(response.additional_kwargs),
                    name=response_model.openai_schema["name"]
                    )
                messages.append(bad_response)
                
                reask_message = ChatMessage(
                    role='user',
                    content = f"Recall the function correctly, fix the errors found:\n\n{e}",
                    additional_kwargs = {'name': response_model.openai_schema["name"]},
                    )
                messages.append(reask_message)

                retries += 1
                
                if retries > max_retries:
                    logger.warning(f"Max retries reached, exception: {e}")
                    raise e
                
                
    def cast(
        self,
        messages: List[BaseMessage],
        response_model: Type[BaseModel] = None,
        max_retries=1,
    ) -> BaseModel:
        """
        Casts text to BaseModel.

        Args:
            messages: The messages containing text to cast to BaseModel.
            response_model: The response model to populate.
            max_retries: The maximum number of retries.

        Returns:
            Validated BaseModel.
            
        ## Usage

        ```python

        class User(BaseModel):
            first_name: str
            age: int
            
        llm = GPT()
        
        messages = [HumanMessage(content="Jason is 20 years old")]

        user = llm.cast(
            messages, 
            response_model=User,
        )                            

        print(user)
        ```
        ## Result

        ```
        User(first_name='Jason', age=20)
            
        ```
        """
        response_model = self._handle_response_model(
            response_model=response_model,
        )  # type: ignore
        response = self._retry_sync(
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
        )  # type: ignore
        return response
    

if __name__ == "__main__":
    client = GPT()