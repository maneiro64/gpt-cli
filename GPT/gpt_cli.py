r"""
Lightweight CLI for prompting ChatGPT via the OpenAI API.

Usage examples:
  - Provide API key via flag:
      python GPT/gpt_cli.py -k YOUR_KEY "Explain recursion in one sentence."
  - Use environment variable:
      setx OPENAI_API_KEY "YOUR_KEY"   (Windows PowerShell)
      python GPT/gpt_cli.py "Write a haiku about rivers."
  - Read prompt from file and stream tokens:
      python GPT/gpt_cli.py -f prompt.txt --stream
  - Pipe from stdin:
      type prompt.txt | python GPT/gpt_cli.py

Config file support (TOML):
  - Default search order: --config path > local GPT/gpt_cli.toml > %APPDATA%\gpt-cli\config.toml > %USERPROFILE%\.gpt-cli.toml
  - Keys: api_key, default_model
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Dict, Any


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gpt-cli",
        description="Prompt ChatGPT (OpenAI) from the command line.",
        add_help=True,
    )

    parser.add_argument(
        "prompt",
        nargs="*",
        help="Prompt text. If omitted, uses --file or reads from stdin.",
    )

    parser.add_argument(
        "-f",
        "--file",
        dest="file_path",
        help="Read prompt from file path.",
    )

    parser.add_argument(
        "-k",
        "--api-key",
        dest="api_key",
        help="OpenAI API key. If omitted, OPENAI_API_KEY env var is used.",
    )

    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help=(
            "Model name. Precedence: CLI > env OPENAI_MODEL > config default_model > 'gpt-5-mini'."
        ),
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config_path",
        default=None,
        help="Path to TOML config file (contains api_key and/or default_model).",
    )

    parser.add_argument(
        "-s",
        "--system",
        dest="system_prompt",
        default=None,
        help="Optional system prompt to steer behavior.",
    )

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens to stdout as they are generated.",
    )

    parser.add_argument(
        "-I",
        "--interactive",
        action="store_true",
        help="Interactive loop mode. Prompts after each response; type 'exit' to quit.",
    )

    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: %(default)s).",
    )

    return parser.parse_args(argv)


def read_prompt_from_args(args: argparse.Namespace) -> str:
    if args.file_path:
        try:
            with open(args.file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except OSError as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(2)

    if args.prompt:
        return " ".join(args.prompt).strip()

    if not sys.stdin.isatty():
        data = sys.stdin.read()
        return data.strip()

    print(
        "No prompt supplied. Provide text, --file, or pipe via stdin. Use -h for help.",
        file=sys.stderr,
    )
    sys.exit(2)


def build_messages(prompt_text: str, system_prompt: Optional[str]) -> List[dict]:
    messages: List[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt_text})
    return messages


def load_toml(path: str) -> Dict[str, Any]:
    try:
        try:
            import tomllib  # type: ignore[attr-defined]
            with open(path, "rb") as f:
                return dict(tomllib.load(f))
        except ModuleNotFoundError:
            import tomli  # type: ignore
            with open(path, "rb") as f:
                return dict(tomli.load(f))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def resolve_config(config_path: Optional[str]) -> Dict[str, Any]:
    if config_path:
        return load_toml(config_path)

    # Common filenames to probe
    filenames = [
        "gpt_cli.toml",
        "gpt-cli.toml",
        "config.toml",
        ".gpt-cli.toml",
    ]

    # Directories to probe in order
    probe_dirs = []

    # 1) Directory of the running executable if frozen (PyInstaller)
    try:
        if getattr(sys, "frozen", False):
            probe_dirs.append(os.path.dirname(sys.executable))
    except Exception:
        pass

    # 2) Current working directory (where the command is invoked)
    probe_dirs.append(os.getcwd())
    # print(f"Current working directory: {os.getcwd()}")

    # 3) The package/script directory (useful for editable installs)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    probe_dirs.append(script_dir)
    # print(f"Script directory: {script_dir}")

    # 4) Support repo layout path `./GPT/gpt_cli.toml` when invoked from repo root
    probe_dirs.append(os.path.join(os.getcwd(), "GPT"))

    # Build candidate paths from probe dirs and filenames
    candidates: List[str] = []
    for d in probe_dirs:
        for name in filenames:
            candidates.append(os.path.join(d, name))

    # 5) Per-user config locations
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(os.path.join(appdata, "gpt-cli", "config.toml"))

    candidates.append(os.path.join(os.path.expanduser("~"), ".gpt-cli.toml"))

    for path in candidates:
        cfg = load_toml(path)
        if cfg:
            return cfg
    return {}


def resolve_model(cli_model: Optional[str], config: Dict[str, Any]) -> str:
    env_model = os.environ.get("OPENAI_MODEL")
    if cli_model:
        return cli_model
    if env_model:
        return env_model
    cfg_model = config.get("default_model") if isinstance(config, dict) else None
    if isinstance(cfg_model, str) and cfg_model.strip():
        return cfg_model.strip()
    return "gpt-5-mini"


def call_openai(
    api_key: Optional[str],
    model: str,
    messages: List[dict],
    stream: bool,
    temperature: float,
) -> int:
    # Lazy import so `-h` works without the package installed
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        print(
            "The 'openai' package is required. Install with: pip install -r requirements.txt",
            file=sys.stderr,
        )
        print(str(e), file=sys.stderr)
        return 3

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def run_with_model(target_model: str) -> int:
        if stream:
            stream_obj = client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            for i, chunk in enumerate(stream_obj):
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None) or ""
                if text:
                    print(text, end="", flush=True)
            print("")
            return 0
        completion = client.chat.completions.create(
            model=target_model,
            messages=messages,
            temperature=temperature,
        )
        text = completion.choices[0].message.content or ""
        print(text)
        return 0

    try:
        return run_with_model(model)
    except Exception as e:
        message = str(e)
        model_not_found = (
            "model_not_found" in message
            or "does not exist" in message
            or "404" in message
        )
        if model_not_found and model != "gpt-4o-mini":
            print(
                f"Selected model '{model}' unavailable. Falling back to 'gpt-4o-mini'.",
                file=sys.stderr,
            )
            try:
                return run_with_model("gpt-4o-mini")
            except Exception as e2:
                print(f"OpenAI API error after fallback: {e2}", file=sys.stderr)
                return 4
        print(f"OpenAI API error: {e}", file=sys.stderr)
        return 4


def call_openai_with_response(
    api_key: Optional[str],
    model: str,
    messages: List[dict],
    stream: bool,
    temperature: float,
) -> tuple[int, str]:
    # Lazy import so `-h` works without the package installed
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        print(
            "The 'openai' package is required. Install with: pip install -r requirements.txt",
            file=sys.stderr,
        )
        print(str(e), file=sys.stderr)
        return 3, ""

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def run_with_model_capture(target_model: str) -> tuple[int, str]:
        if stream:
            accumulated = []
            stream_obj = client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            for i, chunk in enumerate(stream_obj):
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None) or ""
                if text:
                    accumulated.append(text)
                    print(text, end="", flush=True)
            print("")
            return 0, "".join(accumulated)
        completion = client.chat.completions.create(
            model=target_model,
            messages=messages,
            temperature=temperature,
        )
        text = completion.choices[0].message.content or ""
        print(text)
        return 0, text

    try:
        return run_with_model_capture(model)
    except Exception as e:
        message = str(e)
        model_not_found = (
            "model_not_found" in message
            or "does not exist" in message
            or "404" in message
        )
        if model_not_found and model != "gpt-4o-mini":
            print(
                f"Selected model '{model}' unavailable. Falling back to 'gpt-4o-mini'.",
                file=sys.stderr,
            )
            try:
                return run_with_model_capture("gpt-4o-mini")
            except Exception as e2:
                print(f"OpenAI API error after fallback: {e2}", file=sys.stderr)
                return 4, ""
        print(f"OpenAI API error: {e}", file=sys.stderr)
        return 4, ""


def build_context_messages(
    conversation_history: List[dict],
    system_prompt: Optional[str],
) -> List[dict]:
    messages: List[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(conversation_history)
    return messages


def perform_context_request(
    api_key: Optional[str],
    model: str,
    conversation_history: List[dict],
    system_prompt: Optional[str],
    stream: bool,
    temperature: float,
) -> int:
    messages = build_context_messages(conversation_history, system_prompt)
    code, assistant_text = call_openai_with_response(
        api_key=api_key,
        model=model,
        messages=messages,
        stream=stream,
        temperature=temperature,
    )
    if code != 0:
        return code
    conversation_history.append({"role": "assistant", "content": assistant_text})
    return 0


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    config: Dict[str, Any] = resolve_config(args.config_path)

    # API key precedence: CLI --api-key > env OPENAI_API_KEY > config api_key
    effective_api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or (
        config.get("api_key") if isinstance(config, dict) else None
    )
    if effective_api_key:
        os.environ["OPENAI_API_KEY"] = effective_api_key

    model = resolve_model(args.model, config)
    if "gpt-5" in model:
        temperature = 1
    else:
        temperature = float(args.temperature)

    if bool(args.interactive):
        try:
            print("Interactive mode. Type 'exit' to quit.")
            chat_mode = False
            conversation_history: List[dict] = []
            while True:
                try:
                    user_text = input("> ").strip()
                except EOFError:
                    print("")
                    return 0
                except KeyboardInterrupt:
                    print("")
                    return 0
                if user_text.lower() == "exit":
                    return 0
                if not user_text:
                    return 0
                # Handle context commands
                lowered = user_text.lower()
                if lowered.startswith(".context-reset"):
                    chat_mode = True
                    conversation_history = []
                    print("Resetting context mode...")
                    content = user_text[len(".context-reset"):].strip()
                    if not content:
                        continue
                    conversation_history.append({"role": "user", "content": content})
                    code = perform_context_request(
                        api_key=effective_api_key,
                        model=model,
                        conversation_history=conversation_history,
                        system_prompt=args.system_prompt,
                        stream=bool(args.stream),
                        temperature=temperature,
                    )
                    if code != 0:
                        return code
                    continue
                if lowered.startswith(".context-exit"):
                    chat_mode = False
                    conversation_history = []
                    print("Exiting context mode...")
                    content = user_text[len(".context-exit"):].strip()
                    if not content:
                        continue
                    single_turn_messages = build_messages(content, args.system_prompt)
                    return_code = call_openai(
                        api_key=effective_api_key,
                        model=model,
                        messages=single_turn_messages,
                        stream=bool(args.stream),
                        temperature=temperature,
                    )
                    if return_code != 0:
                        return return_code
                    continue
                if lowered.startswith(".context"):
                    chat_mode = True
                    content = user_text[len(".context"):].strip()
                    print(f"Entering context mode...")
                    if not content:
                        continue
                    conversation_history.append({"role": "user", "content": content})
                    code = perform_context_request(
                        api_key=effective_api_key,
                        model=model,
                        conversation_history=conversation_history,
                        system_prompt=args.system_prompt,
                        stream=bool(args.stream),
                        temperature=temperature,
                    )
                    if code != 0:
                        return code
                    continue

                if chat_mode:
                    conversation_history.append({"role": "user", "content": user_text})
                    code = perform_context_request(
                        api_key=effective_api_key,
                        model=model,
                        conversation_history=conversation_history,
                        system_prompt=args.system_prompt,
                        stream=bool(args.stream),
                        temperature=temperature,
                    )
                    if code != 0:
                        return code
                    continue

                # Non-context, single-turn request
                single_turn_messages = build_messages(user_text, args.system_prompt)
                return_code = call_openai(
                    api_key=effective_api_key,
                    model=model,
                    messages=single_turn_messages,
                    stream=bool(args.stream),
                    temperature=temperature,
                )
                if return_code != 0:
                    return return_code
        finally:
            pass

    prompt_text = read_prompt_from_args(args)
    messages = build_messages(prompt_text, args.system_prompt)
    return call_openai(
        api_key=effective_api_key,
        model=model,
        messages=messages,
        stream=bool(args.stream),
        temperature=temperature,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


def run() -> None:
    # Entry point for console_scripts
    sys.exit(main(sys.argv[1:]))

