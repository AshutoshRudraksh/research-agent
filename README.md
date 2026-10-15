# Research agent

Run one topic through `gpt-4o`. The agent searches the web, reads one or two pages, and saves a markdown report under `reports/`.

You need the following on the machine before the first run.

- Python 3.10 or newer. This checkout uses Python 3.11.
- An OpenAI API key in a `.env` file at the repository root. `agent.py` loads that file on startup and binds `ChatOpenAI` to `gpt-4o`.
- A network connection. The model call, `search_web`, and `read_url` all use the public network.

## Create the environment

From the repository root, run these commands.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows Command Prompt, run `venv\Scripts\activate.bat`. In PowerShell, run `.\venv\Scripts\Activate.ps1`.

Put the key in `.env` in the repository root. That file is gitignored.

```
OPENAI_API_KEY=your-key
```

## Research a topic

```bash
python agent.py "quantum computing breakthroughs 2026"
```

The process prints each graph node as it runs. When `write_report` succeeds, the last line is the saved path. `report_path` in `tools.py` builds that path from a slug of the topic and a timestamp with seconds. A second run in the same second gets a numeric suffix, so it does not overwrite the first file.

If the model answers with prose after a successful read, the saved file keeps that prose and appends the page text. The graph allows 8 tool calls. If that cap is hit after a successful read, it still saves the report. If the agent cannot read a page, it writes nothing and exits with status 2.

## Run the tests

The tests do not call OpenAI or the network.

```bash
python -m unittest test_agent.py
```

## Generated files

`write_report` creates `reports/` on first use. That directory is gitignored. Leave the markdown files it writes out of commits.
