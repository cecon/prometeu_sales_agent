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
from langchain.prompts.base import StringPromptTemplate


class CustomPromptTemplateForTools(StringPromptTemplate):
    # The template to use
    template: str
    ############## NEW ######################
    # The list of tools available
    tools_getter: Callable

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        ############## NEW ######################
        tools = self.tools_getter(kwargs["input"])
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        return self.template.format(**kwargs)

