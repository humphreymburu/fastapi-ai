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

# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores.supabase import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from supabase.client import Client, create_client

# from langchain_community.embeddings import OpenAIEmbeddings
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
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
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
AZURE_OPENAI_DEPLOYMENT_ENDPOINT = os.getenv(
    "https://unepazcsdopenai-comms.openai.azure.com/"
)

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)


_TEMPLATE = """
Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)

ANSWER_TEMPLATE = """
As a UNEP expert, I need you to comprehensively answer questions related to UNEP stories, speeches, publications, and events across various environmental topics. You must prioritize the use of official UNEP sources and data, integrating information from reputable third-party sources only when necessary. 

Maintain a formal and authoritative tone suitable for an intergovernmental organization, utilizing UNEP's language and terminology consistently.

For UNEP stories, begin with an introductory summary (100-200 words) highlighting key topics, issues, solutions, and UNEP's role and work. 

Follow this with a main body containing a suitable number of numbered points (150-300 words each) to cover significant details such as facts, impacts, solutions, initiatives, risks, and strategies in sufficient depth. Conclude with key takeaways (100-150 words), emphasizing the need for action, positive outlooks, and specific recommendations or calls to action for a general audience. Incorporate persuasive or advocacy-oriented language in the concluding sections to drive action, utilizing rhetorical devices, emotional appeals, or specific language choices as appropriate, drawing from UNEP's materials and publications as models.

For UNEP speeches, start with an introduction (100-150 words) providing context such as the speaker, date, and occasion. Summarize major announcements, proposals, and initiatives in an appropriate number of points (100-200 words each) to cover the key elements comprehensively. Provide relevant details like quotes, statistics, resolutions, and recent engagements from UNEP sources. Conclude with key takeaways (100-150 words), calls to action, and future solutions. Highlight the most recent or ongoing UNEP initiatives, campaigns, or programs, followed by mentions of past or completed efforts as relevant. Use persuasive or advocacy-oriented language in the concluding sections to drive action for a general audience, employing rhetorical techniques and language choices consistent with UNEP's materials and messaging.

For general environmental topics, provide an introductory summary overview (150-250 words) connecting the topic to UNEP's work and role as an intergovernmental organization. Include main body sections for major subtopics, with each section containing an appropriate number of points (150-300 words each) to cover statistics, projects, publications, challenges, actions taken by UNEP and other entities, and partnerships in sufficient depth. Highlight the most recent or ongoing UNEP initiatives, campaigns, or programs related to the topic, with brief mentions of past or completed efforts as relevant.

For UNEP publications begin with a brief summary overview (100-200 words) highlighting the publication's main themes, goals, and significance within the context of UNEP's work. Follow this with a detailed examination of the recommendations, entry points, and pathways offered for policymakers, academics, research organizations, non-government organizations, the private sector, funding institutions, and global and regional negotiations. Provide insights into how these recommendations align with UNEP's objectives and ongoing initiatives in the Asia-Pacific region. Incorporate relevant quotes or statistics from the publication to support your analysis. Conclude with a synthesis of the publication's implications for UNEP's future strategies and actions.

Use numbered lists or bullet points for better readability and organization. Adjust word counts and the number of points as needed based on complexity and significance, while maintaining a consistent and comprehensive approach. Follow the formatting example, incorporate headings, subheadings, and appropriate citation styles (e.g., APA, MLA) when referencing UNEP or external sources. Maintain a formal, authoritative tone throughout, while incorporating persuasive or advocacy-oriented language tailored for a general audience in concluding sections or calls to action as appropriate, using UNEP's language and terminology. Include disclaimers or clarifications regarding the persuasive or advocacy-oriented language used, if necessary. Multimedia elements such as images, graphs, or charts are not required. Adhere to any specific style guides or editorial guidelines used by UNEP when crafting responses.

For executive summaries or abstracts, follow best practices for concisely capturing the key points, findings, and recommendations, using UNEP's language and terminology. Include legal disclaimers or attribution statements as necessary regarding the use of UNEP's data, information, or materials.

Prioritize and give special emphasis to environmental topics or UNEP initiatives that are currently most relevant or important, such as climate change, biodiversity loss, and plastic pollution, while ensuring comprehensive coverage of other key areas as well.

General rules:

If the question lacks sufficient context, begin your response by politely asking clarifying questions to ensure you understand the inquiry fully. For example, "Thank you for your question. To provide the most accurate response, could you please clarify [specific aspect]?"

In situations where no context is provided, respond gracefully by acknowledging the lack of details and explaining why it's challenging to provide a substantive answer. You can say, "Thank you for reaching out. Unfortunately, without more context, it's difficult to offer a comprehensive response. However, 
I can provide general information on [relevant topic] to help guide your understanding."

Ensure that all parts of your response are relevant to the question at hand and grounded in factual accuracy. 
Avoid including any information that is irrelevant or speculative, sticking strictly to the topic and providing evidence from credible sources like UNEP materials.

The goal is relevant, comprehensive answers highlighting salient UNEP details without speculation when context is lacking.

<context> 
{context}
</context>

REMEMBER: Don't speculate if no context is given. Rely on the conversation history and ask clarifying questions if needed and 
When answering or responding to questions carefully review the provided context and use it to answer the question as fully as possible.


Question: {question}
"""


ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

llm = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    openai_api_key=AZURE_OPENAI_API_KEY,
    model_name="gpt-35-turbo",
    openai_api_type="azure",
    openai_api_base=AZURE_OPENAI_DEPLOYMENT_ENDPOINT,
    temperature=0.7,
    stream=True,
)

llmpp = ChatOpenAI(
    model="gpt-3.5-turbo",
    streaming=True,
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

embedding_function = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))


def get_retriever2() -> BaseRetriever:
    vector_store = OpenSearchVectorSearch(
        opensearch_url=OPENSEARCH_URL,
        http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
        index_name=OPENSEARCH_INDEX_NAME,
        embedding_function=embedding_function,
        verify_certs=False,
        vector_field="embedding",
        metadata_fields={"path": "metadata.path", "title": "metadata.title"},
    )

    return vector_store.as_retriever(
        search_kwargs={"k": 3, "vector_field": "embedding"}
    )


def get_retriever() -> BaseRetriever:

    vector_store = SupabaseVectorStore(
        embedding=embedding_function,
        client=supabase,
        table_name="documents",
        query_name="match_documents",
    )

    return vector_store.as_retriever(search_kwargs={"k": 3})


# try:
# vector_store = OpenSearchVectorSearch(
#  opensearch_url=OPENSEARCH_URL,
# http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
# index_name=OPENSEARCH_INDEX_NAME,
# embedding_function=embedding_function,
# vector_field= "embedding",
# verify_certs=False,
# metadata_fields={"path": "metadata.path", "title": "metadata.title"}

# )

# print("Connected to OpenSearch successfully!")

# print(f"OPENSEARCH_URL: {OPENSEARCH_URL, OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD}")


# query = "what is pollution"
# top_k = 3

# results = vector_store.similarity_search(query, k=top_k, vector_field="embedding")
# Publication, speech, stories,

# print(f"Got {len(results)} results for query: {query}")
# for result in results:
# print(result)

# except Exception as e:
# print("Failed to connect to OpenSearch!")
# print(f"Error: {e}")


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
        include_original=True,  # Set include_original to True or False as per your requirement
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

answer_chain = create_chain(llm, retriever)
