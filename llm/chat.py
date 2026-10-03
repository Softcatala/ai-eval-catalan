#!/usr/bin/env python3
"""Petit xat interactiu de terminal amb llama.cpp, sense dependències externes."""

import argparse
import json
import sys
import urllib.error
import urllib.request
from http.client import HTTPException


DEFAULT_MODEL = "tiny-aya-water-q4_k_m"
HELP = "/nou: esborra la conversa · /ajuda: ajuda · /sortir: surt · Ctrl+C: cancel·la"


def completion_request(args, messages):
    """Build a streaming chat request for llama.cpp."""
    base_url = args.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    payload = {
        "model": args.model,
        "messages": messages,
        "stream": True,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stop": ["<|END_OF_TURN_TOKEN|>"],
    }
    return urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )


def stream_reply(args, messages):
    """Print streamed tokens and return only a successfully completed reply."""
    request = completion_request(args, messages)
    parts = []
    finish_reason = None
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                if not parts:
                    raise RuntimeError("El model ha retornat una resposta buida.")
                return "".join(parts), finish_reason
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(str(event["error"]))
            for choice in event.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    print(content, end="", flush=True)
                    parts.append(content)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    raise RuntimeError("La connexió s'ha tancat abans d'acabar la resposta.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:9090/v1")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--system", default="Ets un assistent útil. Respon en català.")
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.timeout <= 0 or args.temperature < 0:
        parser.error("max-tokens i timeout han de ser positius; temperature, ≥ 0.")
    # Enable line editing and in-memory input history when available.
    try:
        import readline  # noqa: F401
    except ImportError:
        pass

    initial = [{"role": "system", "content": args.system}] if args.system else []
    messages = initial.copy()
    print(f"Xat amb {args.model} · {args.base_url}\n{HELP}\n")
    while True:
        try:
            prompt = input("Tu> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAdeu!")
            return
        if not prompt:
            continue
        if prompt in {"/sortir", "/exit", "/quit"}:
            return
        if prompt in {"/nou", "/clear"}:
            messages = initial.copy()
            print("Conversa esborrada.\n")
            continue
        if prompt in {"/ajuda", "/help"}:
            print(HELP)
            continue

        pending = [*messages, {"role": "user", "content": prompt}]
        print("Model> ", end="", flush=True)
        try:
            reply, reason = stream_reply(args, pending)
        except KeyboardInterrupt:
            print("\n[Resposta cancel·lada; torn descartat.]\n")
            continue
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            print(f"\nError HTTP {error.code}: {detail}", file=sys.stderr)
            print("El torn no s'ha desat. Pots fer /nou si el context és ple.")
            continue
        except (OSError, HTTPException, ValueError, RuntimeError) as error:
            print(f"\nError: {error}", file=sys.stderr)
            print(f"Comprova que llama-server és accessible a {args.base_url}.")
            print("El torn no s'ha desat.\n")
            continue
        messages = [*pending, {"role": "assistant", "content": reply}]
        print()
        if reason == "length":
            print("[Límit de tokens assolit; pots demanar que continuï.]")
        print()


if __name__ == "__main__":
    main()
