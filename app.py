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
from agents import SalesGPT
from chains import SalesConversationChain
from prompts import _conversation_stages, _conversation_stages_text
from templates import CustomPromptTemplateForTools

from tools import get_tools, initialize_procuct_catalog

config = dict(
    salesperson_name = "Julia Goldsmith",
    salesperson_role = "Executivo de Vendas",
    company_name = "Canetas Douradas",
    company_business = "Canetas Douradas é uma empresa de canetas premium que oferece uma gama de canetas de alta qualidade banhadas a ouro. Nossas canetas são projetadas para serem estilosas, funcionais e duradouras, tornando-as perfeitas para profissionais que desejam causar uma impressão duradoura.",
    company_values = "Na Canetas Douradas, acreditamos que a caneta certa pode fazer toda a diferença no mundo. Somos apaixonados por proporcionar aos nossos clientes a melhor experiência de escrita possível e estamos comprometidos com a excelência em tudo o que fazemos.",
    conversation_purpose = "descobrir se o cliente está interessado em adquirir uma caneta premium banhada a ouro.",
    conversation_type = "chat",
    conversation_history = [],
    conversation_stage = _conversation_stages.get('1', "Introdução: Inicie a conversa se apresentando e falando sobre sua empresa. Seja educado e respeitoso, mantendo o tom da conversa profissional."),
)


print(config)
llm = ChatOpenAI(temperature=0.9, verbose=True)
sales_agent = SalesGPT.from_llm(llm, use_tools=True, verbose=True, **config)
# init sales agent
sales_agent.seed_agent()

while True:
    sales_agent.determine_conversation_stage()
    sleep(2)
    sales_agent.step()

    human = input("\nUser Input =>  ")
    if human:
        sales_agent.human_step(human)
        sleep(2)
        print("\n")