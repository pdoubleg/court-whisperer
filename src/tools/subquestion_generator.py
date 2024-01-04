from typing import Tuple, List
from openai import ChatCompletion
from pydantic import Field

from src.utils.pydantic_utils import OpenAISchema


class SubQuestionList(OpenAISchema):
    """List of sub-questions related to a high level question"""
    questions: List[str] = Field(description="Sub-questions related to the main question.")


def generate_subquestions(query: str, n: str) -> List[str]:
    """
    Generate a list of sub-questions from a given query.

    Args:
    query (str): The user query to generate sub-questions from.
    n (str): The range of sub-questions to generate.

    Returns:
    List[str]: A list of generated sub-questions.
    """
    template = f"""
    Your task is to decompose an original user question into {n} distinct sub-questions, \
    such that when they are resolved, the high level question will be answered. \
    Provide these issues separated by newlines. \
    \n\n# Original question:\n{query}
    """

    completion = ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a world class query understanding AI."},
            {"role": "user", "content": template},        
        ],
        functions=[SubQuestionList.openai_schema],
        function_call={"name": SubQuestionList.openai_schema["name"]},
    )

    function_res = SubQuestionList.from_response(completion)
    subquestion_list = function_res.questions

    return subquestion_list