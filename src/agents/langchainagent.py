from dataclasses import dataclass, field
from typing import List
import os

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.base_language import BaseLanguageModel
from langchain.memory import ConversationBufferMemory
from langchain.agents import load_tools
from langchain.chains import LLMMathChain
from langchain import OpenAI, LLMChain
from langchain.tools import StructuredTool
from langchain.utilities import GoogleSearchAPIWrapper, BingSearchAPIWrapper
from ..data.gsuite_integration import *


@dataclass
class AgentConfig:
    google_api_key: str = os.getenv('GOOGLE_API_KEY')
    google_csi_key: str = os.getenv('GOOGLE_CSI_KEY')
    bing_api_key: str = os.getenv('BING_SUBSCRIPTION_KEY')
    bing_api_url: str = os.getenv('BING_SEARCH_URL')

    # google_api_key: str = os.environ['GOOGLE_API_KEY']
    # google_csi_key: str = os.environ['GOOGLE_CSI_KEY']
    # bing_api_key: str = os.environ['BING_SUBSCRIPTION_KEY']
    # bing_api_url: str = "https://api.bing.microsoft.com/bing/v7.0/search"
    prefix: str = """
            Have a conversation with a human, answering the following questions as best you can.
            You have access to the following tools:
        """
    suffix: str = """Begin!"

        {chat_history}
        Question: {input}
        {agent_scratchpad}"""
    memory_key: str = "chat_history"
    input_variables: List[str] = field(default_factory=lambda: ["input", "chat_history", "agent_scratchpad"])


def build_math_tool(llm: BaseLanguageModel):
    math_chain = LLMMathChain.from_llm(llm)
    return Tool(
        name="Math",
        func=math_chain.run,
        description="useful for when you need to do math"
    )


def build_calendar_tool():
    return Tool(
        name="Calendar",
        func=get_calendar_events,
        description="read-only access to Google Calendar, with the ability filter by hours_from_now"
                    " (an integer which can be positive, negative, or zero) used to modify what the time_min of the"
                    " events which are desired"
    )


def build_tools(config: AgentConfig, llm: BaseLanguageModel):
    tools = [
        build_math_tool(llm),
        GmailTool.from_llm(llm),  # NOTE: this LLM is used for summarization, so should probably be a cheap one
        build_calendar_tool()
    ]
    if config.google_api_key and config.google_csi_key:
        google_search = GoogleSearchAPIWrapper()
        tools.append(
            Tool(
                name="Google Search",
                func=google_search.run,
                description="useful for when you need to answer questions about current events"
            )
        )
    if config.bing_api_key:
        bing_search = BingSearchAPIWrapper()
        tools.append(
            Tool(
                name="Bing Search",
                func=bing_search.run,
                description="useful for when you need to search the web using Bing"
            )
        )
    return tools


def build_search_agent(config: AgentConfig):
    llm = OpenAI(temperature=0)
    tools = load_tools(["llm-math"], llm=llm) + build_tools(config, llm=llm)

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=config.prefix,
        suffix=config.suffix,
        input_variables=config.input_variables,
    )
    memory = ConversationBufferMemory(memory_key=config.memory_key)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory
    )
    return agent_chain


# required:
# export GOOGLE_API_KEY=
# export GOOGLE_CSI_KEY=


# export OPENAI_API_KEY=
# export BING_SUBSCRIPTION_KEY=
# export BING_SEARCH_URL=
#
# ...
#
# agent_chain = build_search_agent(AgentConfig())
# agent_chain.run(input="How many people live in the United States?")
