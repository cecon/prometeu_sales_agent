import os
from time import sleep
from typing import Any, Callable, Dict, List, Union
from pydantic import BaseModel, Field
from pathlib import Path
from copy import deepcopy

from langchain import LLMChain, PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.agents import (AgentExecutor, 
                              Tool,
                              LLMSingleActionAgent)
from langchain.llms import BaseLLM
from langchain.chains.base import Chain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

from chains import SalesConversationChain, StageAnalyzerChain
from prompts import _conversation_stages, _conversation_stages_text, _sales_agent_tools_prompt
from templates import CustomPromptTemplateForTools
from langchain.agents import AgentExecutor, create_openai_tools_agent
from tools import get_tools, initialize_procuct_catalog


class SalesGPT(Chain):
    """Controller model for the Sales Agent."""

    stage_analyzer_chain: StageAnalyzerChain = None
    sales_conversation_utterance_chain: SalesConversationChain = None
    sales_agent_executor = Union[AgentExecutor, None]
    tools_names: str = None
    verbose: bool = None
    model_name: str = None
    use_tools:bool = None
    salesperson_name: str = None
    salesperson_role: str = None
    company_name:str = None
    company_business:str = None
    company_values:str = None
    conversation_purpose:str = None
    conversation_type:str = None
    conversation_history = []
    current_conversation_stage: str = None
    conversation_stage_dict: Dict = None

    def __init__(
        self,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(memory=None, verbose=verbose)

        """Inicializa a instância SalesGPT."""
        self.verbose = verbose

        # Atributos preenchidos automaticamente pelo kwargs ou valores padrão
        self.model_name = kwargs.get("model_name", os.getenv("MODEL_NAME"))
        self.use_tools = kwargs.get("use_tools", False)

        self.salesperson_name = kwargs.get("salesperson_name", "Julia Goldsmith")
        self.salesperson_role = kwargs.get("salesperson_role", "Executivo de Vendas")
        self.company_name = kwargs.get("company_name", "Canetas Douradas")
        self.company_business = kwargs.get("company_business", "Canetas Douradas é uma empresa de canetas premium que oferece uma gama de canetas de alta qualidade banhadas a ouro. Nossas canetas são projetadas para serem estilosas, funcionais e duradouras, tornando-as perfeitas para profissionais que desejam causar uma impressão duradoura.")
        self.company_values = kwargs.get("company_values", "Na Canetas Douradas, acreditamos que a caneta certa pode fazer toda a diferença no mundo. Somos apaixonados por proporcionar aos nossos clientes a melhor experiência de escrita possível e estamos comprometidos com a excelência em tudo o que fazemos.")
        self.conversation_purpose = kwargs.get("conversation_purpose", "descobrir se o cliente está interessado em adquirir uma caneta premium banhada a ouro.")
        self.conversation_type = kwargs.get("conversation_type", "chat")

        # Atributos internos da classe
        self.conversation_history = []
        self.current_conversation_stage = '1'
        self.conversation_stage_dict = _conversation_stages

        llm = ChatOpenAI(temperature=0)
        self.stage_analyzer_chain = StageAnalyzerChain.from_llm(llm, verbose=verbose)
        self.sales_conversation_utterance_chain = SalesConversationChain.from_llm(llm, verbose=verbose)

        if "use_tools" in kwargs.keys() and (kwargs["use_tools"] == "True" or kwargs["use_tools"] is True):

            input_variables=[
                "input",
                "tools", 
                "intermediate_steps",
                "salesperson_name",
                "salesperson_role",
                "company_name",
                "company_business",
                "company_values",
                "conversation_purpose",
                "conversation_type",
                "conversation_history",
            ]
            
            tools = get_tools()
            self.tools_names = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

            llm = ChatOpenAI(temperature=0)
            prompt=PromptTemplate(input_variables=input_variables,
                                  template=_sales_agent_tools_prompt
                                  )
            agent = create_openai_tools_agent(llm, tools, prompt)
            self.sales_agent_executor = AgentExecutor(agent=agent, tools=tools)
        else:
            self.sales_agent_executor = None
            self.tools_names = None


    def retrieve_conversation_stage(self, key):
        return self.conversation_stage_dict.get(key, '1')
    
    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    def seed_agent(self):
        # Step 1: seed the conversation
        self.current_conversation_stage= self.retrieve_conversation_stage('1')
        self.conversation_history = []

    def determine_conversation_stage(self):
        print("\n<Calculando o próximo estágio da conversa>:")
        conversation_stage_id = self.stage_analyzer_chain.run(
            conversation_history='"\n"'.join(self.conversation_history), current_conversation_stage=self.current_conversation_stage
        )
        self.current_conversation_stage = self.retrieve_conversation_stage(conversation_stage_id)
        print(f"\n<Conversation Stage>: {self.current_conversation_stage}\n")
        
    def human_step(self, human_input):
        # process human input
        human_input = human_input + '<END_OF_TURN>'
        self.conversation_history.append(human_input)

    def step(self):
        self._call(inputs={})

    def _call(self, inputs: Dict[str, Any]) -> None:
        """Run one step of the sales agent."""

        print("\n<Executando etapa do agente>:")

        # Generate agent's utterance
        inputs = {
            "input": "",
            "tools": self.tools_names,
            "conversation_stage": self.current_conversation_stage,
            "conversation_history": "\n".join(self.conversation_history),
            "salesperson_name": self.salesperson_name,
            "salesperson_role": self.salesperson_role,
            "company_name": self.company_name,
            "company_business": self.company_business,
            "company_values": self.company_values,
            "conversation_purpose": self.conversation_purpose,
            "conversation_type": self.conversation_type,
        }
        if self.use_tools:
            ai_message = self.sales_agent_executor.invoke(inputs)
            output = ai_message["output"]
        else:
            ai_message = self.sales_conversation_utterance_chain.invoke(inputs)
            output = ai_message["text"]
        
        # Add agent's response to conversation history
        self.conversation_history.append(output)

        print(f'\n{self.salesperson_name}: ', output.rstrip('<END_OF_TURN>'))
        return {}

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False, **kwargs) -> "SalesGPT":
        """Initialize the SalesGPT Controller."""

        return cls(
            verbose=verbose,
            **kwargs
        )
