USAGE:

Setup: ensure your environment is active, and your API keys are in the environment:

```bash
$ source venv/bin/activate
$ export OPENAI_API_KEY=...
$ export GOOGLE_API_KEY=...
$ export GOOGLE_CSE_ID=...
```

Run the code in the REPL:

```bash
$ python
>>> from src.agents.langchainagent import build_search_agent
>>> agent = build_search_agent()
>>> agent.run(input="How many active users are there on LinkedIn")
```

