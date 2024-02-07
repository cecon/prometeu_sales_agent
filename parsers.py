import os
from time import sleep
from typing import Any, Callable, Dict, List, Union
from pydantic import BaseModel, Field
from pathlib import Path
import re

from langchain.agents.agent import AgentOutputParser
from langchain.agents.conversational.prompt import FORMAT_INSTRUCTIONS
from langchain.schema import AgentAction, AgentFinish  # OutputParserException
from langchain import LLMChain, PromptTemplate
from langchain.agents import AgentExecutor, Tool
from langchain.chains import LLMChain, RetrievalQA
from langchain.llms import BaseLLM
from langchain.chains.base import Chain
from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


