#!/usr/bin/python

# Copyright (C) 2025 Dory
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import sys
import json
import argparse
import readline
import requests
import base64
import mimetypes
from termcolor import colored, cprint
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule
import itertools
import threading
import time
from .tools import ToolManager


parser = argparse.ArgumentParser(description="OpenAI-compatible chat CLI")
parser.add_argument("--base-url", required=True, help="API base URL")
parser.add_argument("--model", default="", help="Model name")
parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
parser.add_argument("--system", default="", help="System prompt")
parser.add_argument("--hide-thinking", action="store_true")
parser.add_argument("--cache-prompt", action="store_true", help="llama.cpp")
parser.add_argument("--tools-basic", action=argparse.BooleanOptionalAction)
parser.add_argument("--tools-web-access", action=argparse.BooleanOptionalAction)
parser.add_argument("--tools-file-access", action=argparse.BooleanOptionalAction)


def parse_image(user_input):
  parts = user_input.split("@image:")
  if len(parts) > 2 or not user_input.endswith(parts[1]):
    raise ValueError("'@image:' tag must be at the end of the prompt.")

  text_prompt = parts[0].strip()
  image_path = parts[1].strip()
  mime_type, _ = mimetypes.guess_type(image_path)
  if not mime_type or not mime_type.startswith('image/'):
    raise ValueError(f"Error: Unsupported image type '{mime_type}'")

  if image_path.startswith(("http://", "https://")):
    response = requests.get(image_path)
    response.raise_for_status()
    image_blob = response.content

  else:
    if not os.path.exists(image_path):
      raise ValueError(f"Error: Image file not found at '{image_path}'")
    with open(image_path, "rb") as image_file:
      image_blob = image_file.read()

  encoded_image = base64.b64encode(image_blob).decode("utf-8")
  image_url = f"data:{mime_type};base64,{encoded_image}"
  return text_prompt, image_url


def get_model_name(base_url, api_key, model):
  response = requests.get(
      f"{base_url}/models",
      headers={
          "Authorization": f"Bearer {api_key}",
          "Content-Type": "application/json",
      },
  )
  response.raise_for_status()
  models = response.json()

  if not model:
    return models["data"][0]["id"]

  for srv_model in models["data"]:
    if srv_model["id"] == model:
      return srv_model["id"]

  raise ValueError(f"Error: Model '{model}' not found.")


def print_response(console, response, hide_thinking):
  content = response["content"]
  if "reasoning_content" in response and response["reasoning_content"]:
    thinking_text = response["reasoning_content"]
    answer_text = content
  else:
    answer_marker = None
    if "<answer>" in content:
      answer_marker = "<answer>"
    elif "<|end|>" in content:
      answer_marker = "<|end|>"
    elif "</think>" in content:
      answer_marker = "</think>"

    if not answer_marker:
      thinking_text = ""
      answer_text = content
    else:
      parts = content.split(answer_marker, 1)
      thinking_text = parts[0].strip()
      answer_text = parts[1].strip()

  if not hide_thinking and thinking_text:
    cprint(thinking_text + "\n", "magenta")
  console.print(Markdown(answer_text, hyperlinks=False))
  if sys.stdin.isatty():
    print()


def animate(stop_event):
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if stop_event.is_set():
            break
        sys.stdout.write(f'\r{colored(c, "green", attrs=["bold"])} ')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\r')
    sys.stdout.flush()


def call_llm(base_url, api_key, model, messages, cache_prompt, tool_manager):
  json_payload = {
      "model": model, "messages": messages, "stream": False,
  }
  if cache_prompt:
    json_payload["cache_prompt"] = True
  if tool_manager.specs:
    json_payload["tools"] = tool_manager.specs
    json_payload["tool_choice"] = "auto"

  response = requests.post(
      f"{base_url}/chat/completions",
      headers={
          "Authorization": f"Bearer {api_key}",
          "Content-Type": "application/json",
      },
      json=json_payload,
  )
  response.raise_for_status()
  return response.json()["choices"][0]["message"]


def main():
  args = parser.parse_args()
  model_name = get_model_name(args.base_url, args.api_key, args.model)
  if not args.model:
    print(f"Using model: {model_name}", file=sys.stderr)

  tool_manager = ToolManager({
      k.removeprefix("tools_"): v
      for k, v in vars(args).items()
      if k.startswith("tools_")
  })
  if sys.stdin.isatty() and tool_manager.tools:
    print(f"Tools: {', '.join(tool_manager.tools)}")

  messages = []
  if args.system:
    messages.append({"role": "system", "content": args.system})
  console = Console()

  while True:
    try:
      if sys.stdin.isatty():
        console.print(Rule())
        user_input = input(colored("> ", "green", attrs=["bold"]))
        print()
      else:
        user_input = input()

      if "@image:" in user_input:
        try:
          text_prompt, image_url = parse_image(user_input)
        except ValueError as e:
          cprint(e, "red")
          continue
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        })

      else:
        messages.append({"role": "user", "content": user_input})

      if sys.stdin.isatty():
        stop_event = threading.Event()
        animation_thread = threading.Thread(target=animate, args=(stop_event,))
        animation_thread.start()

      try:
        assistant_message = call_llm(
            args.base_url, args.api_key, args.model, messages,
            args.cache_prompt, tool_manager,
        )
        messages.append(assistant_message)

        while tool_manager.tools and assistant_message.get("tool_calls"):
          for tool_call in assistant_message["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])
            tool_result = tool_manager.run_tool(tool_name, **tool_args)
            messages.append({
                "tool_call_id": tool_call["id"],
                "role": "tool",
                "name": tool_name,
                "content": tool_result,
            })

          assistant_message = call_llm(
              args.base_url, args.api_key, args.model, messages,
              args.cache_prompt, tool_manager,
          )
          messages.append(assistant_message)

      finally:
        if sys.stdin.isatty():
          stop_event.set()
          animation_thread.join()

      print_response(console, assistant_message, args.hide_thinking)

    except requests.exceptions.RequestException as e:
      print(f"Error: {e}", file=sys.stderr)
      break
    except (KeyboardInterrupt, EOFError):
      break


if __name__ == "__main__":
  main()
