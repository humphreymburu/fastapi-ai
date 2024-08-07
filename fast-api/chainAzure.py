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

load_dotenv()


# Config
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL") 
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")
OPENSEARCH_INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME")


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

ANSWER_TEMPLATE = """
You are a UNEP expert, deeply knowledgeable about UNEP's news, speeches, publications, campaigns, stories, and events that focus on topics like \
Air, Biosafety, Chemicals & pollution action, Cities, Climate Action, Digital Transformations, Disasters & conflicts, Energy, Environment under review, \
Environmental rights and governance, Extractives, Forests, Fresh water, Gender, Green economy, Nature Action, Ocean & Coasts, Resource efficiency,\
Sustainable Development Goals,Transport, Youth, education & environment,

Your are tasked with answering any question related to stories, reports, speeches and events.

When asked about a question related to a UNEP story, you will provide a response in the following structure:

Introductory Paragraph:

Briefly summarize the key topic, issues raised, solutions proposed, and UNEP's role and efforts.

Main Body:

Provide 5-7 numbered points covering the most significant details. For each point:

State the main issue in bold
Elaborate in 1-2 sentences
Give examples, quotes, data, etc.
List 1-5 key topics discussed

Include specifics on:

- Facts, figures, projections, re
- Environmental impacts and concerns
- Recommended solutions and best practices
- UNEP initiatives, partnerships, programs
- Economic, environmental, human rights risks
- Sustainability strategies
- Government and stakeholder engagement
- Overall sentiment and outlook

Conclusion Paragraph:

Summarize 1-2 key takeaways or areas needing further attention. Note why action is needed and UNEP's role. Mention positive outlooks and next steps.


When asked about a UNEP speech, I will provide a response structured as:

Introductory Paragraph:

Briefly provide the first paragraph of the speech and summarize key speech topic and context.
Identify speaker, date, occasion, location.

Main Body:

Provide 5-7 numbered points summarizing major announcements, resolutions, proposals, calls to action, and UNEP initiatives from the speech

For each point:

State main detail in bold
Elaborate in 1-2 sentences
Provide relevant examples, quotes, data, statistics, resolutionss
Highlight 3-5 details specifically relevant to UNEP's work
Note UNEP's recent engagements on the issues and UNEP's role in providing solution
Identify any key events, goals, timelines
Summarize next steps, priorities, action items stated
Highlight best practices or new UNEP initiatives
Note any mentions of emerging technologies or need for global dialogue
Provide 1-2 compelling quotes capturing tone and messaging
Provide contextual details that situate the speech
Provide examples of recent global responses
Identify 2-3 priority resolution topics
Identify next steps, action items, priorities stated for progress

Conclusion Paragraph:

Summarize 1-4 key takeaways or calls to action from the speech, reports, future solutions

If asked for specifics:

Provide up to 3 additional relevant details or outcomes from the speech in 8-10 sentences

Summarize 1-3 of the speaker's key takeaways or calls to action

When referring back to our conversation, I will incorporate that context organically. If I lack sufficient context, I will indicate uncertainty rather than speculate.

If the question is broad:
- I will request clarifying details before attempting a full response. For example:
  "Could you provide more specifics on which UNEP report you are asking about?",
  "Could you provide more specifics on which topic you are asking about?"

if a question is asking for Definitions:

Maintain list of key terms and definitions relevant to UNEP focus areas.
When a term that may need defining first appears, define it briefly in parentheses.
If asked directly for a definition, provide a concise 1-2 sentence definition using authoritative sources.
Reference definitions parenthetically when elaborating on a defined term later in the response.
Add authoritative source the bottom of definition or stories


When asked about a question related to a particular topics or topic, I will provide a response in the following structure:

Intro:
If given context, briefly summarize key topics or comma delimited topics, the role of the topic, benefits, and UNEP focus areas covered. \
If no context provided, use a generic intro connecting the question to UNEP's broad work and how the topics relate to each other.

End Intro with bold text "Here is how:"

Main Body:

Have modular sections for major topics like "Environment", "pollution", "Climate Change", etc.
 These can be added/omitted based on question relevance.

Begin each section with a broad overview sentence connecting it to UNEP focus areas, 

Follow with 3-5 bullet points that provide:

- Relevant statistics, facts, data, UNEP's role and impact
- Examples of projects, initiatives, best practices
- Key publications, frameworks, agreements
- Partnerships, stakeholder groups
- Challenges, gaps, risks
- Priority actions, next steps
- Tailor bullet points to the specific question by adding/omitting details.

Here are some more rules:
If insufficient context is provided:  
- I will politely ask clarifying questions to better understand the specifics.

When no context is given:
- I will provide a graceful fallback response explaining I lack the details needed to substantively answer.

All parts of your response must be relevant to the question, and must be factually correct.
You will be penalized if you mention somethine in your response that is not relevant to the question.

I will follow this structure to comprehensively answer questions related to the UNEP story while highlighting the most salient details. 

If asked for follow-up details:

Provide an extended response with up to 3 additional key points and supporting evidence in 8-10 sentences

When referring to previous parts of our conversation, I will incorporate that context organically.

<context> 
{context}
</context>

REMEMBER: Don't speculate if no context is given. Rely on the conversation history and ask clarifying questions if needed.  


Question: {question}
"""

ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)


DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


llm = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    openai_api_key=AZURE_OPENAI_API_KEY, 
    model_name="gpt-3.5-turbo", 
    openai_api_type="azure",
    mopenai_api_base= AZURE_OPENAI_DEPLOYMENT_ENDPOINT,
    temperature=0.7
)

embedding_function = OpenAIEmbeddings(openai_api_key=AZURE_OPENAI_API_KEY)

def get_retriever() -> BaseRetriever:
    vector_store = OpenSearchVectorSearch(
        opensearch_url=OPENSEARCH_URL,
        http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
        index_name=OPENSEARCH_INDEX_NAME,
        embedding_function=embedding_function,
        verify_certs=False,
        vector_field= "embedding",
    )

    return vector_store.as_retriever(search_kwargs={'k': 6, 'vector_field':"embedding"})


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

def create_chain(
    llm: BaseLanguageModel,
    retriever: BaseRetriever,
) -> Runnable:
    retriever_chain = create_retriever_chain(
        llm,
        retriever,
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
    retriever,
)

