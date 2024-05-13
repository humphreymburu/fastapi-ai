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
from typing import Dict, List, Optional, Sequence
from typing import AsyncIterator

from fastapi import FastAPI
from langchain.schema import Document
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain_community.chat_models import ChatOpenAI
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.runnables.base import (
    Runnable,
    RunnableBinding,
    RunnableGenerator,
    RunnableLambda,
    RunnableMap,
    RunnableParallel,
    RunnableSequence,
    RunnableSerializable,
)
from langchain.schema.messages import AIMessage, HumanMessage
from langchain_core.runnables.branch import RunnableBranch
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.router import RouterInput, RouterRunnable
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.opensearch_vector_search import (
    OpenSearchVectorSearch,
)
from langchain.prompts import (ChatPromptTemplate, MessagesPlaceholder,
                               PromptTemplate)
from langchain_core.retrievers import BaseRetriever
from langchain.schema.language_model import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser

from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain.cache import InMemoryCache
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.globals import set_llm_cache
from langchain.globals import set_debug

set_debug(True)


load_dotenv()

set_llm_cache(InMemoryCache())


# Config
OPENSEARCH_URL = os.getenv("v2_OPENSEARCH_URL") 
OPENSEARCH_USERNAME = os.getenv("v2_OPENSEARCH_USERNAME")
OPENSEARCH_PASSWORD = os.getenv("v2_OPENSEARCH_PASSWORD")
OPENSEARCH_INDEX_NAME = os.getenv("v2_OPENSEARCH_INDEX_NAME")

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") 
AZURE_OPENAI_DEPLOYMENT_ENDPOINT = os.getenv("https://unepazcsdopenai-comms.openai.azure.com/")


_TEMPLATE = """
Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)

def load_answer_template(file):
    template_folder ="templates"
    file_path = os.path.join(template_folder, file)
    with open(file_path, "r") as file:
        answer_template = file.read()
    return answer_template

ANSWER_TEMPLATE = load_answer_template("answer.txt")

ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

llm = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    openai_api_key=AZURE_OPENAI_API_KEY, 
    model_name="gpt-35-turbo", 
    openai_api_type="azure",
    openai_api_base= AZURE_OPENAI_DEPLOYMENT_ENDPOINT,
    temperature=0.7,
    stream=True,
)

llmpp = ChatOpenAI(
    model="gpt-3.5-turbo",
    streaming=True,
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

embedding_function = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

def get_retriever() -> BaseRetriever:
    vector_store = OpenSearchVectorSearch(
        opensearch_url=OPENSEARCH_URL,
        http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
        index_name=OPENSEARCH_INDEX_NAME,
        embedding_function=embedding_function,
        verify_certs=False,
        vector_field= "embedding",
        metadata_fields={"path": "metadata.path", "title": "metadata.title"}
    )

    return vector_store.as_retriever(search_kwargs={'k': 3, 'vector_field':"embedding"})
    

#try:
    #vector_store = OpenSearchVectorSearch(
      #  opensearch_url=OPENSEARCH_URL,
        #http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
        #index_name=OPENSEARCH_INDEX_NAME,
        #embedding_function=embedding_function,
        #vector_field= "embedding",
        #verify_certs=False,
        #metadata_fields={"path": "metadata.path", "title": "metadata.title"}

   # )
    
    #print("Connected to OpenSearch successfully!")

    #print(f"OPENSEARCH_URL: {OPENSEARCH_URL, OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD}")


    #query = "what is pollution"
    #top_k = 3

    #results = vector_store.similarity_search(query, k=top_k, vector_field="embedding")
    #Publication, speech, stories, 

    #print(f"Got {len(results)} results for query: {query}")
    #for result in results: 
       #print(result)

#except Exception as e:
  #print("Failed to connect to OpenSearch!")
    #print(f"Error: {e}")


# User input
class ChatHistory(BaseModel):
    """Chat history with the bot."""
    question: str
    chat_history: Optional[List[Dict[str, str]]]

def create_retriever_chain(
    llm: BaseLanguageModel, retriever: BaseRetriever
) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)
    condense_question_chain = (
        CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )
    conversation_chain = condense_question_chain | retriever
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
            RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            )
            | retriever
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")

def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)

def serialize_history(request: ChatHistory):
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

def create_chain(
    llm: BaseLanguageModel,
    retriever: BaseRetriever,

) -> Runnable:
        
    # Create a MultiQueryRetriever instance
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
        include_original=True  # Set include_original to True or False as per your requirement
    )

    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=multi_query_retriever
    )

    # Integrate MultiQueryRetriever into the chain
    retriever_chain = create_retriever_chain(
        llm,
        compression_retriever,  # Use MultiQueryRetriever instead of the base retriever
    ).with_config(run_name="FindDocs")


    _context = RunnableMap(
        {
            "context": retriever_chain | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
    ).with_config(run_name="RetrieveDocs")

    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    response_synthesizer = (prompt | llm | StrOutputParser()).with_config(
        run_name="GenerateResponse",
    )
    return (
        
        {
            "question": RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            ),
            "chat_history": RunnableLambda(serialize_history).with_config(
                run_name="SerializeHistory"
            ),
        }
        | _context
        | response_synthesizer
        
    )



retriever = get_retriever()

answer_chain = create_chain(
    llm,
    retriever
)

