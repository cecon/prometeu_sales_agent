# https://github.com/PromptEngineer48/Sales_Agent_using_LangChain/blob/main/sales_pen_git.py

import os
from time import sleep
from typing import Any, Callable, Dict, List, Union
from pydantic import BaseModel, Field
from pathlib import Path

from langchain import LLMChain, PromptTemplate
from langchain.agents import (AgentExecutor, 
                              Tool,
                              LLMSingleActionAgent)
from langchain.chains import LLMChain, RetrievalQA
from langchain.llms import BaseLLM
from langchain.chains.base import Chain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from prompts import (_stage_analyzer_inception_prompt_template,
                    _sales_agent_inception_prompt,
                    _conversation_stages, 
                    _conversation_stages_text)
from templates import CustomPromptTemplateForTools

from tools import get_tools, initialize_procuct_catalog


class StageAnalyzerChain(LLMChain):
    """Chain to analyze which conversation stage should the conversation move into."""

    def __init__(self, prompt: PromptTemplate, llm: BaseLLM, verbose: bool = True):
            """
            Inicializa a instância de StageAnalyzerChain.
            """
            super().__init__(prompt=prompt, llm=llm, verbose=verbose)

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        ## The above class method returns an instance of the LLMChain class.

        ## The StageAnalyzerChain class is designed to be used as a tool for analyzing which 
        ## conversation stage should the conversation move into. It does this by generating 
        ## responses to prompts that ask the user to select the next stage of the conversation 
        ## based on the conversation history.
        """Get the response parser."""
        prompt = PromptTemplate(
            template=_stage_analyzer_inception_prompt_template,
            input_variables=["conversation_history"],
        )

        return cls(prompt=prompt, llm=llm, verbose=verbose)


class SalesConversationChain(LLMChain):
    """Chain to generate the next utterance for the conversation."""

    def __init__(self, prompt: PromptTemplate, llm: BaseLLM, verbose: bool = True):
            """
            Inicializa a instância de SalesConversationChain.
            """
            super().__init__(prompt=prompt, llm=llm, verbose=verbose)

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""

        sales_agent_inception_prompt = _sales_agent_inception_prompt

        prompt = PromptTemplate(
            template=sales_agent_inception_prompt,
            input_variables=[
                "salesperson_name",
                "salesperson_role",
                "company_name",
                "company_business",
                "company_values",
                "conversation_purpose",
                "conversation_type",
                "conversation_stage",
                "conversation_history"
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

