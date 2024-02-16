#!/usr/bin/env python
"""Example LangChain server exposes a conversational retrieval chain.

Follow the reference here:

https://python.langchain.com/docs/expression_language/cookbook/retrieval#conversational-retrieval-chain

To run this example, you will need to install the following packages:
pip install langchain openai faiss-cpu tiktoken
"""  # noqa: F401

from operator import itemgetter
from typing import List, Tuple
import os
from dotenv import load_dotenv


from fastapi import FastAPI
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain_community.chat_models import ChatOpenAI
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.opensearch_vector_search import (
    OpenSearchVectorSearch,
)

from langchain_core.output_parsers import StrOutputParser

from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field

load_dotenv()


# Config
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL") 
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")
OPENSEARCH_INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME")


UNEP_EXPERT_TEMPLATE = """
You are an expert on UNEP, the United Nations Environment Programme. You have comprehensive knowledge of UNEP's mission, activities, publications, and news stories spanning climate change, pollution, nature protection, and more.

Your expertise covers the full range of UNEP's work, including major initiatives like:
- The #BeatPollution strategy and campaign
- Report launches like the Air Quality in Asia and Clean Air for Blue Skies report
- Climate and Clean Air Coalition programs and science-based pollution/climate solutions
- Hosting international environmental events and partnerships

You can discuss and answer questions about any UNEP publication, announcement, or news story in depth. Your answers will demonstrate comprehensive knowledge and insights into UNEP's environmental efforts globally.

UNEP Article: {context}

Question: {question}

Answer: 
"""


_TEMPLATE = """Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)

ANSWER_TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)












DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    streaming=True,
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)


embedding_function = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

vector_store = OpenSearchVectorSearch(
        opensearch_url=OPENSEARCH_URL,
        http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
        index_name=OPENSEARCH_INDEX_NAME,
        embedding_function=embedding_function,
        verify_certs=False,
        vector_field= "embedding",
)

retriever = vector_store.as_retriever(search_kwargs={'k': 3, 'vector_field':"embedding"})




try:
    vector_store = OpenSearchVectorSearch(
        opensearch_url=OPENSEARCH_URL,
        http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
        index_name=OPENSEARCH_INDEX_NAME,
        embedding_function=embedding_function,
        vector_field= "embedding",
        verify_certs=False
    )
    
    print("Connected to OpenSearch successfully!")

    print(f"OPENSEARCH_URL: {OPENSEARCH_URL, OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD}")


    query = "any news related to recent UN Environment Programme (UNEP) report on air quality?"
    top_k = 3

    #results = vector_store.similarity_search(query, k=top_k, vector_field="embedding")

    #print(f"Got {len(results)} results for query: {query}")
    #for result in results: 
        #print(result)


except Exception as e:
    print("Failed to connect to OpenSearch!")
    print(f"Error: {e}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    """Combine documents into a single string."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple]) -> str:
    """Format chat history into a string."""
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer


_inputs = RunnableMap(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(temperature=0)
    | StrOutputParser(),
)
_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}



# User input
class ChatHistory(BaseModel):
    """Chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    question: str


conversational_qa_chain = (
    _inputs | _context | ANSWER_PROMPT | ChatOpenAI() | StrOutputParser()
)
chain = conversational_qa_chain.with_types(input_type=ChatHistory)