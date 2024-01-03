import argparse

from src.agents.langchainagent import AgentConfig, build_search_agent


def build_and_run_search_agent(query: str):
    agent = build_search_agent(AgentConfig())
    agent.run(query)


def search_cli():
    parser = argparse.ArgumentParser(
        prog='Smithers AI',
        description='This is a description of our cool AI'
    )
    parser.add_argument('search', type=str,
                        help='something you would like to look up')
    args = parser.parse_args()
    build_and_run_search_agent(args.search)
