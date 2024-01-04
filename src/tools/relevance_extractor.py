import logging
from typing import Optional, no_type_check
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from src.parsing.utils import (
    extract_numbered_segments, 
    number_segments, 
    clean_whitespace, 
    get_specs,
    )


logger = logging.getLogger(__name__)


def extract_relevant_passages(query: Optional[str] = None, passage: Optional[str] = None, segment_length: int = 1):
    model = ChatOpenAI(model="gpt-4-1106-preview", temperature=0.0)
    # clean up extra whitespace
    passage_clean = clean_whitespace(passage)
    # number the segments in the passage
    numbered_passage = number_segments(passage_clean, segment_length)
    # build prompt to extract number tags from relevant lines
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "The user will give you a PASSAGE containing segments numbered as <#1#>, <#2#>, <#3#>, etc., followed by a QUERY. Extract ONLY the segment-numbers from the PASSAGE that are RELEVANT to the QUERY.",
        ),
        (
            "human",
            "PASSAGE:\n{numbered_passage}\n\nQUERY:\n{query}",
        ),
    ]
    )

    chain = LLMChain(llm=model, prompt=prompt)
    spec = chain.run(query=query, numbered_passage=numbered_passage)
    specs = get_specs(spec)
    extracts = extract_numbered_segments(numbered_passage, specs)
    return extracts
        
