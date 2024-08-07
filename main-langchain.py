#!/usr/bin/env python
"""Example LangChain server exposes a conversational retrieval chain.

Follow the reference here:

https://python.langchain.com/docs/expression_language/cookbook/retrieval#conversational-retrieval-chain

To run this example, you will need to install the following packages:
pip install langchain openai faiss-cpu tiktoken
"""  # noqa: F401

from operator import itemgetter
from typing import List, Tuple
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from langchain.schema import format_document
from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

# from chain import ChatHistory, chain
##from pydantic import BaseModel

from chain import ChatHistory, answer_chain

app = FastAPI(
    title="Generative Chat",
    version="1.0",
    description="UNEP RAG AI API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_credentials=True,
    # allow_methods=["*"],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream

add_routes(
    app,
    answer_chain,
    enable_feedback_endpoint=True,
    path="/chat",
    input_type=ChatHistory,
    config_keys=["metadata"],
)


# add_routes(app, chain, enable_feedback_endpoint=True)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)

    import time
