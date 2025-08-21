## gpt-cli (Windows PowerShell)

Minimal Python CLI to prompt ChatGPT via OpenAI's API.

### Setup (PowerShell)

1) Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install -r GPT/requirements.txt
```

### Configure API key

You can provide the API key via flag, environment variable, or config file (TOML).

- One-off flag

```powershell
python GPT/gpt_cli.py -k YOUR_OPENAI_API_KEY "Hello"
```

- Persistent environment variable

```powershell
setx OPENAI_API_KEY "YOUR_OPENAI_API_KEY"
# Open a new PowerShell after setting it
```

- Config file (TOML)

Create one of the following (searched in order):

1. Same folder as the executable or current working directory: `gpt_cli.toml`, `gpt-cli.toml`, or `config.toml`
2. Script/package directory (editable installs): `GPT/gpt_cli.toml`
3. `%APPDATA%\gpt-cli\config.toml`
4. `%USERPROFILE%\.gpt-cli.toml`

Example content:

```toml
api_key = "YOUR_OPENAI_API_KEY"
default_model = "gpt5-fast"
```

You can also pass an explicit path:

```powershell
python GPT/gpt_cli.py --config C:\path\to\config.toml "Your prompt"
```

### Usage

```powershell
python GPT/gpt_cli.py -h
python GPT/gpt_cli.py "Explain recursion in one sentence."
python GPT/gpt_cli.py -f prompt.txt --stream
type prompt.txt | python GPT/gpt_cli.py -s "You are terse." -m gpt5-fast -t 0.2
```

### Install as a command (Windows)

Build and install so `gpt-cli` is on PATH:

```powershell
# From repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip build
python -m build
pip install .\dist\gpt-cli-0.1.0-py3-none-any.whl

# Verify
gpt-cli -h
```

If `gpt-cli` is not found, ensure your Python Scripts directory is on PATH. Common locations:

```powershell
$env:Path += ";$env:USERPROFILE\AppData\Local\Programs\Python\Python311\Scripts"
```

### Defaults and precedence

- Default model: `gpt5-fast`
- Model override precedence: `--model` > `OPENAI_MODEL` env var > `default_model` in config > default
- API key precedence: `--api-key` > `OPENAI_API_KEY` env var > `api_key` in config
- Temperature default: `0.7`
- Reads prompt from positional args, `--file`, or stdin


