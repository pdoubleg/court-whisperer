import ast
import hashlib
import logging
import os
import sys
from dotenv import load_dotenv
import warnings
import traceback
from enum import Enum
from functools import cache
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    no_type_check,
)

from httpx import Timeout
import openai
from pydantic import BaseModel

from src.language_models.base import (
    LanguageModel,
    LLMConfig,
    LLMFunctionCall,
    LLMFunctionSpec,
    LLMMessage,
    LLMResponse,
    LLMTokenUsage,
    Role,
)

from src.language_models.utils import (
    async_retry_with_exponential_backoff,
    retry_with_exponential_backoff,
)

logging.getLogger("openai").setLevel(logging.ERROR)

NO_ANSWER="I-DONT-KNOW"


def friendly_error(e: Exception, msg: str = "An error occurred.") -> str:
    tb = traceback.format_exc()
    original_error_message: str = str(e)
    full_error_message: str = (
        f"{msg}\nOriginal error: {original_error_message}\nTraceback:\n{tb}"
    )
    return full_error_message


class OpenAIChatModel(str, Enum):
    """Enum for OpenAI Chat models"""
    GPT3_5_TURBO = "gpt-3.5-turbo-1106"
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-1106-preview"
    
_context_length: Dict[str, int] = {
    # can add other non-openAI models here
    OpenAIChatModel.GPT3_5_TURBO: 4096,
    OpenAIChatModel.GPT4: 8192,
    OpenAIChatModel.GPT4_TURBO: 128_000,
}

_cost_per_1k_tokens: Dict[str, Tuple[float, float]] = {
    # can add other non-openAI models here.
    # model => (prompt cost, generation cost) in USD
    OpenAIChatModel.GPT3_5_TURBO: (0.0015, 0.002),
    OpenAIChatModel.GPT4: (0.03, 0.06),  # 8K context
    OpenAIChatModel.GPT4_TURBO: (0.01, 0.03),  # 128K context
}



class OpenAIGPTConfig(LLMConfig):
    type: str = "openai"
    max_output_tokens: int = 1024
    min_output_tokens: int = 64
    timeout: int = 20
    temperature: float = 0.2
    chat_model: OpenAIChatModel = OpenAIChatModel.GPT4
    context_length: Dict[str, int] = {
        OpenAIChatModel.GPT3_5_TURBO: 4096,
        OpenAIChatModel.GPT4: 8192,
        OpenAIChatModel.GPT4_TURBO: 128_000,
    }
    cost_per_1k_tokens: Dict[str, Tuple[float, float]] = {
        # (input/prompt cost, output/completion cost)
        OpenAIChatModel.GPT3_5_TURBO: (0.0015, 0.002),
        OpenAIChatModel.GPT4: (0.03, 0.06),  # 8K context
        OpenAIChatModel.GPT4_TURBO: (0.01, 0.03),
    }

class OpenAIResponse(BaseModel):
    """OpenAI response model, either completion or chat."""
    choices: List[Dict]  # type: ignore
    usage: Dict  # type: ignore


# Define a class for OpenAI GPT-3 that extends the base class
class OpenAIGPT(LanguageModel):
    """
    Class for OpenAI LLMs
    """

    def __init__(self, config: OpenAIGPTConfig):
        """
        Args:
            config: configuration for openai-gpt model
        """
        super().__init__(config)
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.api_base: str | None = os.getenv("OPENAI_API_BASE")
        if self.api_key == "":
            raise ValueError(
                """
                OPENAI_API_KEY not set in .env file,
                please set it to your OpenAI API key."""
            )
            
    def chat_context_length(self) -> int:
        """
        Context-length for chat-completion models/endpoints
        """
        model = (
            self.config.chat_model
        )
        return _context_length.get(model, super().chat_context_length())
            
    def chat_cost(self) -> Tuple[float, float]:
        """
        (Prompt, Generation) cost per 1000 tokens, for chat-completion
        models/endpoints.
        Get it from the dict, otherwise fail-over to general method
        """
        return _cost_per_1k_tokens.get(self.config.chat_model, super().chat_cost())
            
    def _cost_chat_model(self, prompt: int, completion: int) -> float:
        price = self.chat_cost()
        return (price[0] * prompt + price[1] * completion) / 1000

    def _get_non_stream_token_usage(
        self, response: Dict[str, Any]
    ) -> LLMTokenUsage:
        """
        Extracts token usage from ``response`` and computes cost.
        """
        cost = 0.0
        prompt_tokens = 0
        completion_tokens = 0
        prompt_tokens = response["usage"]["prompt_tokens"]
        completion_tokens = response["usage"]["completion_tokens"]
        cost = self._cost_chat_model(
            response["usage"]["prompt_tokens"],
            response["usage"]["completion_tokens"],
        )

        return LLMTokenUsage(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost=cost
        )

    def generate(self, prompt: str, max_tokens: int) -> LLMResponse:
            try:
                return self._generate(prompt, max_tokens)
            except Exception as e:
                # capture exceptions not handled by retry, so we don't crash
                err_msg = str(e)[:500]
                logging.error(f"OpenAI API error: {err_msg}")
                return LLMResponse(message=NO_ANSWER, cached=False)

    def _generate(self, prompt: str, max_tokens: int) -> LLMResponse:
        return self.chat(messages=prompt, max_tokens=max_tokens)


    async def agenerate(self, prompt: str, max_tokens: int) -> LLMResponse:
        try:
            return await self._agenerate(prompt, max_tokens)
        except Exception as e:
            # capture exceptions not handled by retry, so we don't crash
            err_msg = str(e)[:500]
            logging.error(f"OpenAI API error: {err_msg}")
            return LLMResponse(message=NO_ANSWER, cached=False)

    async def _agenerate(self, prompt: str, max_tokens: int) -> LLMResponse:
        openai.api_key = self.api_key
        if self.api_base:
            openai.api_base = self.api_base
        messages = [
            LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
            LLMMessage(role=Role.USER, content=prompt),
        ]

        @async_retry_with_exponential_backoff
        async def completions_with_backoff(
            **kwargs: Dict[str, Any]
        ) -> str:
            result = await openai.ChatCompletion.acreate(  # type: ignore
                    **kwargs
                )

            return result

            response = await completions_with_backoff(
                model=self.config.chat_model,
                messages=[m.api_dict() for m in messages],
                max_tokens=max_tokens,
                request_timeout=self.config.timeout,
                temperature=self.config.temperature,
            )
            msg = response["choices"][0]["message"]["content"].strip()

            @retry_with_exponential_backoff
            async def completions_with_backoff(**kwargs):  # type: ignore
                result = await openai.Completion.acreate(**kwargs)  # type: ignore
                return result

            response = await completions_with_backoff(
                model=self.config.completion_model,
                prompt=prompt,
                max_tokens=max_tokens,
                request_timeout=self.config.timeout,
                temperature=self.config.temperature,
            )
            msg = response["choices"][0]["text"].strip()
        return LLMResponse(message=msg)

    def chat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int,
        functions: Optional[List[LLMFunctionSpec]] = None,
        function_call: str | Dict[str, str] = "auto",
    ) -> LLMResponse:
        try:
            return self._chat(messages, max_tokens, functions, function_call)
        except Exception as e:
            # capture exceptions not handled by retry, so we don't crash
            err_msg = str(e)[:500]
            logging.error(f"OpenAI API error: {err_msg}")
            return LLMResponse(message=NO_ANSWER)

    def _chat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int,
        functions: Optional[List[LLMFunctionSpec]] = None,
        function_call: str | Dict[str, str] = "auto",
    ) -> LLMResponse:
        """
        ChatCompletion API call to OpenAI.
        Args:
            messages: list of messages  to send to the API, typically
                represents back and forth dialogue between user and LLM, but could
                also include "function"-role messages. If messages is a string,
                it is assumed to be a user message.
            max_tokens: max output tokens to generate
            functions: list of LLMFunction specs available to the LLM, to possibly
                use in its response
            function_call: controls how the LLM uses `functions`:
                - "auto": LLM decides whether to use `functions` or not,
                - "none": LLM blocked from using any function
                - a dict of {"name": "function_name"} which forces the LLM to use
                    the specified function.
        Returns:
            LLMResponse object
        """
        openai.api_key = self.api_key
        if self.api_base:
            openai.api_base = self.api_base
        if isinstance(messages, str):
            llm_messages = [
                LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
                LLMMessage(role=Role.USER, content=messages),
            ]
        else:
            llm_messages = messages

        @retry_with_exponential_backoff
        def completions_with_backoff(**kwargs):  # type: ignore
            result = openai.ChatCompletion.create(**kwargs)  # type: ignore
            return result

        # Azure uses different parameters. It uses ``engine`` instead of ``model``
        # and the value should be the deployment_name not ``self.config.chat_model``
        chat_model = self.config.chat_model
        key_name = "model"
        if self.config.type == "azure":
            key_name = "engine"
            if hasattr(self, "deployment_name"):
                chat_model = self.deployment_name

        args: Dict[str, Any] = dict(
            **{key_name: chat_model},
            messages=[m.api_dict() for m in llm_messages],
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=self.config.temperature,
            request_timeout=self.config.timeout,
        )
        # only include functions-related args if functions are provided
        # since the OpenAI API will throw an error if `functions` is None or []
        if functions is not None:
            args.update(
                dict(
                    functions=[f.dict() for f in functions],
                    function_call=function_call,
                )
            )
        response = completions_with_backoff(**args)

        # openAI response will look like this:
        """
        {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "name": "", 
                    "content": "\n\nHello there, how may I help you?",
                    "function_call": {
                        "name": "fun_name",
                        "arguments: {
                            "arg1": "val1",
                            "arg2": "val2"
                        }
                    }, 
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        }
        """
        message = response["choices"][0]["message"]
        msg = message["content"] or ""
        if message.get("function_call") is None:
            fun_call = None
        else:
            fun_call = LLMFunctionCall(name=message["function_call"]["name"])
            try:
                fun_args = ast.literal_eval(message["function_call"]["arguments"])
                fun_call.arguments = fun_args
            except (ValueError, SyntaxError):
                logging.warning(
                    "Could not parse function arguments: "
                    f"{message['function_call']['arguments']} "
                    f"for function {message['function_call']['name']} "
                    "treating as normal non-function message"
                )
                fun_call = None
                msg = message["content"] + message["function_call"]["arguments"]

        return LLMResponse(
            message=msg.strip() if msg is not None else "",
            function_call=fun_call,
            usage=self._get_non_stream_token_usage(response),
        )
