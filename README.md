# Smithers AI
## Setup
### To setup your environment you will have to run the following commands
#### create the virtual environment
```shell
python -m venv venv
```
#### Then use one of the following commands compatible with your operating system to activate the python virtual environment
mac/linux:
```shell
source ./venv/bin/activate
```
windows powershell:
```shell
./venv/bin/activate.ps1
```
#### Finally run install
```shell
pip install -r requirements.txt
```

### Next create a .env file in the project's root directory for the API keys used for the toolchain
#### The file contents should look like the following where <some_key> is the API key from that provider
```env
OPENAI_API_KEY=<some_key>
GOOGLE_API_KEY=<some_key>
GOOGLE_CSI_KEY=<some_key>
BING_SUBSCRIPTION_KEY=<some_key>
BING_SEARCH_URL=https://api.bing.microsoft.com/bing/v7.0/search
```

## Run
### Once installed you can run smithers using the following command
```shell
smithers
What would you like to ask? (type "exit" to quit). Go!
What's my name?
[chain/start] [1:chain:AgentExecutor] Entering Chain run with input:
...
```
