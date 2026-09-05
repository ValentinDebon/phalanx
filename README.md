# Phalanx

Command line interface for a [Ministral 3 8B Instruct](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512).
Based on the llama.cpp C API for inference, it supports tool calling
by invoking external processes and standard input/output redirection.

## Why?

For some use-cases, running a local LLM to answer trivial
questions can be really useful. Even more when you are offline.
But the old knowledge database (circa 2023 for this model), can
be a limiting factor.

If I wanted function calling or MCP, I could have used `llama-server`
as my primary user interface. However I also use my main computer for
gaming. And taking several gigabytes of video memory would require me
to turn the server off and on again. So I prefer using `llama-cli`,
knowing the model loads relatively fast (~1 second). The other reason
was that I wanted to decouple the tools from the inference, and most
solutions take either megabytes of runtime, or a ridiculous
amount of script plumbing to configure.

## How does it work?

The interface is a wrapper around the LLM inference.
It detects the tokens introducing function calls/arguments.
It spawns external tools, feeding them the generated arguments
and then feeding the LLM back with the generated output.
Note the tokens are **hard coded** so only the given model
is the sole supported, even though all Mistral models
support the same chat template.

The [model](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-GGUF/blob/main/Ministral-3-8B-Instruct-2512-Q8_0.gguf)
is not provided with the repository. I use the 8-bits quantized version.
It provides a solid 30+ tokens per second on ARC A770 (16GB) with the Intel Xe driver,
using the Mesa ANV Vulkan backend of llama.cpp. All running on Debian testing.

## How to use it

The model should be downloaded and available at the
configured large language model path, which defaults to
`$(datadir)/Ministral-3-8B-Instruct-2512-Q8_0.gguf`.

The tools are looked for in `$XDG_DATA_HOME/phalanx`, and the
tools list in `$XDG_CONFIG_HOME/phalanx/available_tools.json`.

```
phalanx # Starts the prompt, no history supported, getline-based.
```

## Build

First, install its dependencies, on a Debian-based distribution:
```sh
sudo apt install libllama-dev
```

Download a source release package and extract it locally.

Then, configure, build and install:
```sh
./configure
make install
```

## Copying

Phalanx sources, binaries and documentations are distributed under the Affero GNU Public License version 3.0, see LICENSE.
