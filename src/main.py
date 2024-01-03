import argparse

from src.agents.langchainagent import AgentConfig, build_search_agent


def build_and_run_search_agent(query: str):
    agent = build_search_agent(AgentConfig())
    agent.run(query)


def smithers_loop():
    line = ''
    while True:
        print('What would you like to ask?')
        line = input()
        if line == 'exit'.strip().casefold():
            break
        build_and_run_search_agent(line.strip())


def search_cli():
    parser = argparse.ArgumentParser(
        prog='Smithers AI',
        description='This is a description of our cool AI'
    )
    # parser.add_argument('search', type=str,
    #                     help='something you would like to look up')
    # args = parser.parse_args()
    smithers_loop()
