import argparse
from dotenv import load_dotenv

from src.agents.langchainagent import AgentConfig, build_agent


def run_agent(agent, query: str):
    agent.run(query)


def smithers_loop(agent):
    line = ''
    while True:
        print('What would you like to ask? (type "exit" to quit). Go!')
        line = input()
        if line == 'exit'.strip().casefold():
            break
        run_agent(agent, line.strip())


def search_cli():
    parser = argparse.ArgumentParser(
        prog='Smithers AI',
        description='This is a description of our cool AI'
    )
    # parser.add_argument('search', type=str,
    #                     help='something you would like to look up')
    # args = parser.parse_args()
    load_dotenv()
    agent = build_agent(AgentConfig())
    smithers_loop(agent)
