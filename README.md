# RiCA: Multi-threaded Reasoning for Large Language Models

[![Run in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rica-team/rica-server/blob/main/demo.ipynb)

**RiCA** is a framework that enables Large Language Models (LLMs) to perform multi-threaded reasoning. It allows LLMs to call external tools, continue thinking while waiting for the results, and handle multiple tool calls concurrently.

## Key Features

- **Multi-threaded Reasoning**: LLMs can execute long-running tasks in the background without blocking their thought process.
- **Tool Integration**: Easily integrate external tools and APIs through a simple decorator-based routing system.
- **Real-time Interaction**: Observe the model's "thinking" process in real-time through token-by-token streaming.
- **Flexible and Extensible**: Designed to be adapter-based, allowing for easy integration with different LLM backends (e.g., Transformers, OpenAI).

## Getting Started

### Installation

You can install RiCA and its dependencies using pip:

```bash
pip install .[pt]
```

### Running the Demo

The `demo/example.py` script provides a simple example of how to use RiCA. To run it, first make sure you have the necessary dependencies installed, then run the following command from the project root:

```bash
PYTHONPATH=. python demo/example.py
```

You can also explore the `demo.ipynb` notebook for a more interactive experience.

## How it Works

RiCA works by introducing a "multi-threaded thinking" capability to LLMs. When an LLM needs to use an external tool, it can choose to run it in the background. While the tool is running, the LLM can continue to reason, process information, and even call other tools. When a background tool finishes, its result is seamlessly inserted back into the LLM's context, allowing it to incorporate the new information into its reasoning process.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.
