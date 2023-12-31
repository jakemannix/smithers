from dataclasses import dataclass, field
from typing import List, Type
import os

from langchain.agents import ZeroShotAgent, Tool, initialize_agent, AgentType, AgentExecutor
from pydantic import BaseModel, Field, BaseSettings
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.agents import load_tools
from langchain.chains import LLMMathChain, ConversationChain
from langchain.globals import set_debug
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.callbacks import FileCallbackHandler
from loguru import logger
from langchain.utilities import GoogleSearchAPIWrapper, BingSearchAPIWrapper


from ..data.gsuite_integration import *


@dataclass
class AgentConfig:
    serper_api_key: str = os.getenv('SERPER_API_KEY')
    google_api_key: str = os.getenv('GOOGLE_API_KEY')
    google_csi_key: str = os.getenv('GOOGLE_CSI_KEY')
    bing_api_key: str = os.getenv('BING_SUBSCRIPTION_KEY')
    bing_api_url: str = os.getenv('BING_SEARCH_URL')

    debug: bool = bool(os.getenv("DEBUG", "False"))

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
    logfile: str = "agent.log"
    llm_temperature: float = 0.0


def build_math_tool(llm: BaseLanguageModel):
    math_chain = LLMMathChain.from_llm(llm)
    return Tool(
        name="Math",
        func=math_chain.run,
        description="useful for when you need to do math"
    )


def build_time_tool():
    datetime_tool = Tool(
        name="Datetime",
        func=lambda x: datetime.now().isoformat(),
        description="Returns the current datetime",
    )
    return datetime_tool


class GCalToolSchema(BaseModel):
    time_min: str = Field(default=None,
                          description="string, Lower bound for an event's end time to filter by. Optional (if not "
                                      "specified, defaults to current time). "
                                      "The default is not to filter by end time. Must be an RFC3339 timestamp with "
                                      "mandatory time zone offset, for example, 2011-06-03T10:00:00-07:00 or "
                                      "2011-06-03T10:00:00Z")
    time_max: str = Field(default=None,
                          description="string, Upper bound for an event's start time to filter by. Optional. "
                                      "The default is not to filter by start time. Must be an RFC3339 timestamp with "
                                      "mandatory time zone offset")
    max_results: int = Field(default=10, description="an integer specifying the maximum number of results to return")
    fields: str = Field(default="*", description=
                        """
    a comma-separated list of fields to return, or * for all fields. Available fields (and sub-fields) are:
    ["kind", "etag", "id", "status", "htmlLink", "created", "updated", "summary", "creator(email,displayName,self)",
     "organizer(email,displayName,self)", "attendees(email,displayName,responseStatus,self)", "location", "description",
     "start(date,dateTime,timeZone)", "end(date,dateTime,timeZone)",
     "transparency", "iCalUID", "sequence", "reminders(useDefault,overrides(method,minutes))", "eventType"].
                        """)
    calendar_id: str = Field(default='primary', description="the calendar ID to query")


class GCalTool(BaseTool, BaseSettings):
    name: str = "gcal"
    description: str = "read-only access to Google Calendar, with the ability filter by time range (via time_min and" \
                       " time_max parameters).  MAKE SURE TO USE the 'fields' parameter to reduce the response size" \
                       " by only returning the fields you need.  e.g. \"fields\": \"summary,creator(email,self)\" "
    args_schema: Type[GCalToolSchema] = GCalToolSchema

    def _run(
        self,
        time_min: str = None,
        time_max: str = None,
        max_results: int = 10,
        fields: str = "*",
        calendar_id: str = "primary"
    ) -> Any:
        return get_calendar_events(time_min=time_min,
                                   time_max=time_max,
                                   max_results=max_results,
                                   fields="items(" + fields + ")",
                                   calendar_id=calendar_id)

    async def _arun(self,
                    time_min: str,
                    time_max: str,
                    max_results: int,
                    fields: str,
                    calendar_id: str,
                    **kwargs: Any,
                    ) -> Any:
        raise NotImplementedError("gcal does not support async")


def build_calendar_tool():
    return GCalTool()


def build_tools(config: AgentConfig, llm: BaseLanguageModel):
    tools = [
        build_math_tool(llm),
        GmailTool.from_llm(llm),  # NOTE: this LLM is used for summarization, so should probably be a cheap one
        build_calendar_tool(),
        build_time_tool()
    ]
    if config.serper_api_key:
        # need to figure out how to make sure this works
        search = GoogleSerperAPIWrapper(serper_api_key=config.serper_api_key)
        tools.append(
            Tool(
                name="Google Search",
                func=search.run,
                description="useful for when you need to search the web using Google"
            )
        )
    if config.google_api_key and config.google_csi_key:
        google_search = GoogleSearchAPIWrapper()
        tools.append(
            Tool(
                name="Google Search",
                func=google_search.run,
                description="useful for when you need to answer questions about current events"
            )
        )
    if config.bing_api_key and False:
        bing_search = BingSearchAPIWrapper()
        tools.append(
            Tool(
                name="Bing Search",
                func=bing_search.run,
                description="useful for when you need to search the web using Bing"
            )
        )
    return tools


def build_agent(config: AgentConfig) -> AgentExecutor:
    logger.add(config.logfile, colorize=False, enqueue=True)
    set_debug(config.debug)
    handler = FileCallbackHandler(config.logfile)

    llm = ChatOpenAI(model="gpt-3.5-turbo-16k",
                     temperature=config.llm_temperature,
                     verbose=True)
    tools = load_tools(["llm-math"], llm=llm) + build_tools(config, llm=llm)

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=config.prefix,
        suffix=config.suffix,
        input_variables=config.input_variables,
    )
    chat_history = MessagesPlaceholder(variable_name="chat_history")
    memory = ConversationBufferMemory(llm=llm, memory_key=config.memory_key,
                                      return_messages=True, max_token_length=4096)
    # conversation = ConversationChain(memory=memory, llm=llm, verbose=config.debug)
    # llm_chain = LLMChain(llm=llm, prompt=prompt)
    # agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain: AgentExecutor = initialize_agent(tools=tools,
                                                  llm=llm,
                                                  agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                                                  memory=memory,
                                                  verbose=True,
                                                  agent_kwargs={
                                                      "memory_prompts": [chat_history],
                                                      "input_variables": config.input_variables
                                                  },
                                                  callbacks=[handler])
    agent_chain.agent.llm_chain.verbose = True
    # agent_chain = AgentExecutor.from_agent_and_tools(
    #     agent=agent, tools=tools, verbose=True, memory=memory, callbacks=[handler]
    # )
    return agent_chain
