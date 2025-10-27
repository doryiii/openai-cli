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

import collections
import inspect
import os
import json
import requests
import html2text
from datetime import datetime
from pydantic import Field
from termcolor import cprint
from typing import Annotated


class Basic:
  def get_time():
    """Get the current local date, time, and timezone."""
    now = datetime.now()
    return f"{now.strftime("%A")} {now.isoformat()} {now.astimezone().tzinfo}"


class WebAccess:
  def web_fetch(
      url: Annotated[str, Field(description="the webpage URL to fetch")],
  ):
    """Get content of a webpage."""
    if not url.startswith(("http://", "https://")):
      url = "https://" + url
    webres = requests.get(url)
    webres.raise_for_status()
    return html2text.html2text(webres.text)

  def web_search(
      query: Annotated[str, Field(description="the web search query")],
      num_results: Annotated[int, Field(description="how many pages to get. Default 5")] = 5,
  ):
    """Search the web."""
    res = requests.post(
        "https://api.langsearch.com/v1/web-search",
        json={"query": str(query), "summary": True, "count": int(num_results)},
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get("LANGSEARCH_API_KEY")}",
        },
    ).json()
    cleaned_res = [
        {
            "name": pg["name"],
            "url": pg["url"],
            "summary": pg["summary"] or pg["snippet"],
        }
        for pg in res["data"]["webPages"]["value"]
    ]
    return json.dumps(cleaned_res)


class FileAccess:
  def list_dir(
      path: Annotated[str, Field(description="the directory to list")],
  ):
    """List content of a directory."""
    return "\n".join(os.listdir(path))

  def read_file(path: Annotated[str, Field(description="the file to read")]):
    """Read content of a file."""
    with open(path) as f:
      return f.read()

  def write_file(
      path: Annotated[str, Field(description="the file to write to")],
      content: Annotated[str, Field(description="the content to write")],
  ):
    """Write content to a file."""
    with open(path, "w") as f:
      f.write(content)
    return "Done"


TOOL_TYPE = {"basic": Basic, "web_access": WebAccess, "file_access": FileAccess}


class ToolManager:
  TYPES = collections.defaultdict(lambda: "string", {int: "integer"})

  def __init__(self, tool_types):
    enabled_tool_classes = [TOOL_TYPE[t] for t in tool_types if tool_types[t]]
    self._tools = {}
    self.specs = []
    for tool_class in enabled_tool_classes:
      for tool_name, tool_func in tool_class.__dict__.items():
        if not tool_name.startswith("_") and callable(tool_func):
          self._tools[tool_name] = tool_func
          self.specs.append(self._get_spec(tool_func))

  def _get_spec(self, f):
    spec = {
        "type": "function",
        "function": {
            "name": f.__name__,
            "description": inspect.getdoc(f),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    for name, p in inspect.signature(f).parameters.items():
      if p.annotation is not inspect.Parameter.empty:
        # The type hint is in the first arg of Annotated
        param_type = self.TYPES[p.annotation.__args__[0]]
        # The description is in the `description` field of the second arg
        param_desc = p.annotation.__metadata__[0].description
        spec["function"]["parameters"]["properties"][name] = {
            "type": param_type,
            "description": param_desc,
        }
        if p.default is inspect.Parameter.empty:
          spec["function"]["parameters"]["required"].append(name)

    if not spec["function"]["parameters"]["properties"]:
      spec["function"]["parameters"] = {}

    return spec

  def call(self, tool_name, **kwargs):
    if tool_name not in self._tools:
      raise ValueError(f"Tool '{tool_name}' not found.")
    args_text = ", ".join(f"{k}='{v}'" for k, v in kwargs.items())
    cprint(f"{tool_name}({args_text})", "magenta")
    return self._tools[tool_name](**kwargs)

