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

from langchain.schema import format_document
from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field
from chain import ChatHistory, chain


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)
# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream


add_routes(app, chain, enable_feedback_endpoint=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)