"""
Interactive prompt sender for the Kimodo OSC server.

Usage:
    python prompt_sender.py

Type a prompt, press Enter. The script blocks until the server finishes
generating (and has swapped the new motion into the live stream), then
prints a short summary and asks for the next prompt. Ctrl+C to quit.

Optional inline overrides, space-separated after the prompt:
    a person doing jumping jacks --duration 6 --seed 42 --steps 80

Recognized flags: --duration, --seed, --steps, --fps
"""

import requests
import shlex
import sys

SERVER_URL = "http://127.0.0.1:42069/generate"


def parse_prompt_line(line: str):
    """Splits a line into (prompt_text, overrides_dict).
    Any --flag value pairs are pulled out; everything else is the prompt."""
    tokens = shlex.split(line)
    overrides = {}
    prompt_tokens = []

    flag_map = {
        "--duration": ("duration", float),
        "--seed": ("seed", int),
        "--steps": ("steps", int),
        "--fps": ("fps", int),
    }

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in flag_map and i + 1 < len(tokens):
            key, caster = flag_map[tok]
            try:
                overrides[key] = caster(tokens[i + 1])
            except ValueError:
                print(f"  (!) Couldn't parse value for {tok}, ignoring it.")
            i += 2
        else:
            prompt_tokens.append(tok)
            i += 1

    return " ".join(prompt_tokens).strip(), overrides


def send_prompt(prompt: str, overrides: dict):
    payload = {
        "prompt": prompt,
        "filename_prefix": "live",
    }
    payload.update(overrides)

    print(f"  -> Sending: \"{prompt}\"" + (f"  {overrides}" if overrides else ""))
    print("  -> Generating (this will block until done)...")

    try:
        response = requests.post(SERVER_URL, json=payload, timeout=600)
    except requests.exceptions.ConnectionError:
        print("  (!) Could not reach the server. Is it running on port 42069?")
        return
    except requests.exceptions.Timeout:
        print("  (!) Request timed out after 10 minutes.")
        return

    if response.status_code != 200:
        print(f"  (!) Server returned an error [{response.status_code}]: {response.text}")
        return

    data = response.json()
    metrics = data.get("performance_metrics", {})
    params = data.get("executed_parameters", {})
    print(
        f"  Done. inference={metrics.get('inference_duration_seconds', '?')}s, "
        f"total={metrics.get('total_request_handling_seconds', '?')}s, "
        f"frames={params.get('frames', '?')} @ {params.get('fps', '?')}fps"
    )


def main():
    print("=" * 60)
    print(" Kimodo prompt sender — type a prompt and press Enter.")
    print(" Add flags if needed: --duration 6 --seed 42 --steps 80")
    print(" Ctrl+C to quit.")
    print("=" * 60)

    while True:
        try:
            line = input("\nPrompt> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

        if not line:
            continue

        prompt, overrides = parse_prompt_line(line)
        if not prompt:
            print("  (!) Empty prompt after removing flags, skipping.")
            continue

        send_prompt(prompt, overrides)


if __name__ == "__main__":
    main()