import json
from typing import List, Optional, Union

from pydantic import BaseModel, Extra

from src.language_models.base import (
    LLMFunctionCall,
    LLMMessage,
    LLMResponse,
    LLMTokenUsage,
    Role,
)
from src.types import DocMetaData, Document
from src.parsing.json import extract_top_level_json, top_level_json_field


class ChatDocAttachment(BaseModel):
    # any additional data that should be attached to the document
    class Config:
        extra = Extra.allow


class ChatDocMetaData(DocMetaData):
    usage: Optional[LLMTokenUsage]


class ChatDocLoggerFields(BaseModel):
    tool_type: str = ""
    tool: str = ""
    content: str = ""

    @classmethod
    def tsv_header(cls) -> str:
        field_names = cls().dict().keys()
        return "\t".join(field_names)


class ChatDocument(Document):
    function_call: Optional[LLMFunctionCall] = None
    metadata: ChatDocMetaData
    attachment: None | ChatDocAttachment = None

    def __str__(self) -> str:
        fields = self.log_fields()
        tool_str = ""
        if fields.tool_type != "":
            tool_str = f"{fields.tool_type}[{fields.tool}]: "
        recipient_str = ""
        if fields.recipient != "":
            recipient_str = f"=>{fields.recipient}: "
        return (
            f"{fields.sender_entity}[{fields.sender_name}] "
            f"{recipient_str}{tool_str}{fields.content}"
        )

    def get_json_tools(self) -> List[str]:
        """
        Get names of attempted JSON tool usages in the content
            of the message.
        Returns:
            List[str]: list of JSON tool names
        """
        jsons = extract_top_level_json(self.content)
        tools = []
        for j in jsons:
            json_data = json.loads(j)
            tool = json_data.get("request")
            if tool is not None:
                tools.append(tool)
        return tools

    def log_fields(self) -> ChatDocLoggerFields:
        """
        Fields for logging in csv/tsv logger
        Returns:
            List[str]: list of fields
        """
        tool_type = ""  # FUNC or TOOL
        tool = ""  # tool name or function name
        if self.function_call is not None:
            tool_type = "FUNC"
            tool = self.function_call.name
        elif self.get_json_tools() != []:
            tool_type = "TOOL"
            tool = self.get_json_tools()[0]
        content = self.content
        if tool_type == "FUNC":
            content += str(self.function_call)
        return ChatDocLoggerFields(
            tool_type=tool_type,
            tool=tool,
            content=content,
        )

    def tsv_str(self) -> str:
        fields = self.log_fields()
        fields.content = shorten_text(fields.content, 80)
        field_values = fields.dict().values()
        return "\t".join(str(v) for v in field_values)

    def pop_tool_ids(self) -> None:
        """
        Pop the last tool_id from the stack of tool_ids.
        """
        if len(self.metadata.tool_ids) > 0:
            self.metadata.tool_ids.pop()

    @staticmethod
    def from_LLMResponse(
        response: LLMResponse,
    ) -> "ChatDocument":
        """
        Convert LLMResponse to ChatDocument.
        Args:
            response (LLMResponse): LLMResponse to convert.
        Returns:
            ChatDocument: ChatDocument representation of this LLMResponse.
        """
        message = response.message
        return ChatDocument(
            content=message,
            function_call=response.function_call,
            metadata=ChatDocMetaData(
                usage=response.usage,
            ),
        )


ChatDocMetaData.update_forward_refs()