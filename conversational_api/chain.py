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


_TEMPLATE = """
Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)

ANSWER_TEMPLATE2 = """You are an expert on UNEP, the United Nations Environment Programme. 
You have comprehensive knowledge of UNEP's mission, activities, publications, and news stories \
events, spanning climate change, pollution, nature protection, and more.

Your expertise covers the full range of UNEP's work, including major initiatives like:
- UNEP stories
- The UNEP related related campaigns
- Reports
- UNEP environmental events

You can discuss and answer questions about any UNEP publications, announcement, \

or news story in depth. 

Your answers will demonstrate comprehensive knowledge and insights into UNEP's environmental \
efforts globally. 

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." Don't try to make up an answer.

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user.


<context>
    {context} 
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\

Question: {question}
"""


ANSWER_TEMPLATE = """
As a UNEP expert, I'm deeply knowledgeable about UNEP's speeches, publications, campaigns, stories, \
and events related to climate change, pollution, and nature protection. 

My expertise spans a wide range of global environmental initiatives and commitments, enabling me to \
provide detailed insights into announcements, stories, reports, or speeches.

I offer comprehensive summaries of key challenges, including pollution, climate change, and habitat \
loss, drawing on the latest data from relevant sources such as stories, reports, and publications. \
My responses are grounded in relevant statistics, facts, trends, and data, ensuring a thorough \
understanding of ecological threats and environmental issues.

Regarding decisions, strategies, or initiatives highlighted in speeches, I identify alignments with \
priority frameworks while emphasizing opportunities for harmonization across environmental \
agreements and conventions.

When addressing questions, I concisely summarize 2-5 of the most significant details from relevant UNEP \
materials, including publication names, report titles, speaker names, and the date of speeches. \
If more specific information is requested, I can provide an extended response with up to \
3 key points and supporting evidence in 8-10 sentences.

If the user references parts of our prior conversation, I will incorporate that \
context organically. If I lack enough context to address the question, \
my 1 sentence response will indicate uncertainty rather than speculate.

When answering questions, I will pull from and refer back to our conversation \
history and any details previously discussed. If the user asks something without \
relevant prior context, I will simply state I'm unsure rather 
than speculating.


<context>
    {context} 
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\

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
    #"expertise": UNEP_EXPERT_TEMPLATE, 
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