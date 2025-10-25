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

import enum
import functools
import os
import json
import requests
import html2text
from abc import ABC, abstractmethod
from datetime import datetime
from termcolor import cprint


class ToolType(enum.Enum):
  BASIC = "basic"
  FILE_ACCESS = "file_access"
  WEB_ACCESS = "web_access"


class Tools(ABC):
  @property
  @abstractmethod
  def spec(self):
    ...

  @property
  @abstractmethod
  def tool_type(self):
    ...

  @abstractmethod
  def run(self, **kwargs):
    ...


class ToolManager:
  def __init__(self, tool_types):
    enabled_tool_types = {ToolType(t) for t in tool_types if tool_types[t]}
    self.tools = {}
    self.specs = []
    for tool_class in Tools.__subclasses__():
      try:
        tool_instance = tool_class()
        if tool_instance.tool_type not in enabled_tool_types:
          continue
        tool_name = tool_instance.spec["function"]["name"]
        self.tools[tool_name] = tool_instance
        self.specs.append(tool_instance.spec)
      except ValueError as e:
        print(f"Cannot initialize {tool_class.__name__}: {e}")

  def run_tool(self, tool_name, **kwargs):
    if tool_name in self.tools:
      args_text = ", ".join(f"{k}='{v}'" for k, v in kwargs.items())
      cprint(f"{tool_name}({args_text})", "magenta")
      return self.tools[tool_name].run(**kwargs)
    else:
      raise ValueError(f"Tool '{tool_name}' not found.")


class Time(Tools):
  @functools.cached_property
  def spec(self):
    return {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current local time.",
            "parameters": {},
        },
    }

  @functools.cached_property
  def tool_type(self):
    return ToolType.BASIC

  def run(self):
    now = datetime.now()
    return f"{now.strftime("%A")} {now.isoformat()} {now.astimezone().tzinfo}"

class WebFetchTool(Tools):
  @functools.cached_property
  def spec(self):
    return {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Get the content of a webpage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the webpage to fetch.",
                    }
                },
                "required": ["url"],
            },
        },
    }

  @functools.cached_property
  def tool_type(self):
    return ToolType.WEB_ACCESS

  def run(self, url):
    try:
      response = requests.get(url)
      response.raise_for_status()
      return html2text.html2text(response.text)
    except requests.exceptions.RequestException as e:
      return f"Error: {e}"


class WebSearchTool(Tools):
  def __init__(self):
    if not os.environ.get('LANGSEARCH_API_KEY'):
      raise ValueError("LANGSEARCH_API_KEY not set")

  @functools.cached_property
  def spec(self):
    return {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Performs a web search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "The number of search results to return.",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    }

  @functools.cached_property
  def tool_type(self):
    return ToolType.WEB_ACCESS

  def run(self, query, num_results=3):
    try:
      response = requests.post(
          "https://api.langsearch.com/v1/web-search",
          headers={
              "Authorization": f"Bearer {os.environ.get('LANGSEARCH_API_KEY')}",
              "Content-Type": "application/json",
          },
          json={"query": query, "summary": True, "count": num_results},
      )
      response.raise_for_status()
      cleaned_response = [
          {
              "name": pg["name"], "url": pg["url"],
              "summary": pg["summary"] or pg["snippet"],
          }
          for pg in response.json()["data"]["webPages"]["value"]
      ]
      return json.dumps(cleaned_response)
    except requests.exceptions.RequestException as e:
      return f"Error: {e}"


class ListDir(Tools):
  @functools.cached_property
  def spec(self):
    return {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "Lists the contents of a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the directory.",
                    }
                },
                "required": ["path"],
            },
        },
    }

  @functools.cached_property
  def tool_type(self):
    return ToolType.FILE_ACCESS

  def run(self, path):
    try:
      return "\n".join(os.listdir(path))
    except Exception as e:
      return f"Error: {e}"


class ReadFile(Tools):
  @functools.cached_property
  def spec(self):
    return {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file.",
                    }
                },
                "required": ["path"],
            },
        },
    }

  @functools.cached_property
  def tool_type(self):
    return ToolType.FILE_ACCESS

  def run(self, path):
    try:
      with open(path, "r") as f:
        return f.read()
    except Exception as e:
      return f"Error: {e}"


class WriteFile(Tools):
  @functools.cached_property
  def spec(self):
    return {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Writes content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    }

  @functools.cached_property
  def tool_type(self):
    return ToolType.FILE_ACCESS

  def run(self, path, content):
    try:
      with open(path, "w") as f:
        f.write(content)
      return f"Successfully wrote to {path}"
    except Exception as e:
      return f"Error: {e}"
