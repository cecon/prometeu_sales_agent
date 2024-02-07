# https://github.com/PromptEngineer48/Sales_Agent_using_LangChain/blob/main/sales_pen_git.py

import os
from time import sleep
from typing import Any, Callable, Dict, List, Union
from pydantic import BaseModel, Field
from pathlib import Path

from langchain import LLMChain, PromptTemplate
from langchain.agents import AgentExecutor, Tool, tool
from langchain.chains import LLMChain, RetrievalQA
from langchain.llms import BaseLLM
from langchain.chains.base import Chain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.tools.retriever import create_retriever_tool
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma, Qdrant, FAISS
from langchain_community.document_loaders import TextLoader


def initialize_procuct_catalog():
    filename = "product_catalog.txt"
    product_catalog = Path(__file__).parent / filename
    
    model_name = os.getenv("MODEL_NAME")
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    embeddings = OpenAIEmbeddings()
    text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    loader = TextLoader(product_catalog)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    return retriever

#@tool
def product_search1(query: str) -> str:
    """útil para quando você precisa responder a perguntas sobre informações sobre os produtos"""
    knowledge_base = initialize_procuct_catalog()
    result1 = knowledge_base.run(query)
    result2 = knowledge_base.get_relevant_docs(query)
    return result1


def get_tools():
    knowledge_base = initialize_procuct_catalog()
    tool = create_retriever_tool(
        knowledge_base,
        "catalogo_de_produtos",
        "útil para quando você precisa responder a perguntas ou recuperar informações sobre produtos.",
    )
    tools = [tool]
    return tools