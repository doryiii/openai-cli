# Yet Another OpenAI-compatible CLI

A simple yet feature-rich command-line chat client compatible with the OpenAI
API.

*(or "I just want to quickly test my model hosted with vllm/llama.cpp but don't
want to spin up openwebui")*

![Chat CLI](./chat.png)

## Features

- **OpenAI API Compatible:** Works with any self-hosted LLM platform that
  supports OpenAI chat completions API.
- **Image Support:** Send local or remote images to models that support vision.
- **Tools:** The script has built-in tools for getting the time, fetching web
  pages, performing web searches, and file reading/writing.
- **Syntax Highlighting:** Renders Markdown in the terminal for better
  readability.
- **Extremely simple:** Single file, no installation needed.

## Installation

- From AUR
  ```bash
  yay -S yaoc-git
  ```
- From PyPI
  ```bash
  pip install yaoc
  ```
- From Github
  ```bash
  git clone https://github.com/doryiii/yaoc.git
  cd yaoc
  pip install -r requirements.txt
  ```
  (Or use your operating system's package manager to install python requests,
  termcolor, rich, and html2text.)

## Usage

 - If installed as system package
   ```bash
   openai-cli --base-url "http://localhost:8090/v1"
   ```
 - If installed with PyPI
   ```bash
   python -m yaoc.openai_cli --base-url "http://localhost:8090/v1"
   ```
 - If cloned from Github
   ```bash
   python src/yaoc/openai_cli.py --base-url "http://localhost:8090/v1"
   ```

### Flags

| Flag              | Description                                                                         |
| ----------------- | ----------------------------------------------------------------------------------- |
| `--base-url`      | **(Required)** The base URL of the OpenAI-compatible API.                           |
| `--model`         | The model to use. If not given, the first model returned by the API will be used.   |
| `--api-key`       | The API key for the service. Defaults to the `OPENAI_API_KEY` environment variable. |
| `--system`        | Optional system prompt to give to the model                                         |
| `--hide-thinking` | Hide the thinking process output of the model.                                      |
| `--cache-prompt`  | llama.cpp specific prompt caching                                                   |
| `--tools-basic`   | Enable basic tools (e.g. get_time)                                                  |
| `--tools-web-access`  | Enable web search and fetch tools                                                   |
| `--tools-file-access` | Enable file system tools                                                            |

### Images

![Image Example](./image.png)

To send an image, use the `@image:` tag at the end of your prompt:

```
> Tell me about this image @image:path/to/your/image.png
```

You can also use a URL:

```
> What do you see in this image? @image:https://example.com/image.jpg
```

### Piping

You can pipe text into stdin:

```bash
echo "Hello!" | python src/openai_cli/main.py --base-url "http://localhost:8090/v1" --hide-thinking
```

### Tools

The script has some built-in tools:

- `get_time`: Get the current local time.
- `web_fetch`: Get the content of a webpage.
- `web_search`: Performs a web search. Requires the `LANGSEARCH_API_KEY`
  environment variable to be set.
- `list_dir`: Lists the contents of a directory.
- `read_file`: Reads the contents of a file.
- `write_file`: Writes content to a file.

## Dependencies

- [requests](https://pypi.org/project/requests/)
- [termcolor](https://pypi.org/project/termcolor/)
- [rich](https://pypi.org/project/rich/)
- [html2text](https://pypi.org/project/html2text/)

