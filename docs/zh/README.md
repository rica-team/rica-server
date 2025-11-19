# RiCA：大型语言模型的多线程推理框架

| [English](../../README.md) | [中文](README.md) |
| :--- | :--- |

[![Run in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rica-team/rica-server/blob/main/demo.ipynb)

**RiCA** 是一个赋能大型语言模型（LLM）进行多线程推理的框架。它允许 LLM 调用外部工具，在等待结果的同时继续思考，并并发处理多个工具调用。

## 主要特性

- **多线程推理**：LLM 可以在后台执行耗时任务，而不会阻塞其思维过程。
- **工具集成**：通过简单的基于装饰器的路由系统，轻松集成外部工具和 API。
- **实时交互**：通过逐 token 流式传输，实时观察模型的“思考”过程。
- **灵活且可扩展**：采用基于适配器的设计，易于集成不同的 LLM 后端（例如 Transformers, OpenAI）。

## 快速开始

### 安装

您可以使用 pip 安装 RiCA 及其依赖项：

```bash
pip install .[pt]
```

### 运行演示

`demo/example.py` 脚本提供了一个简单的 RiCA 使用示例。要运行它，请首先确保已安装必要的依赖项，然后在项目根目录下运行以下命令：

```bash
PYTHONPATH=. python demo/example.py
```

您也可以探索 `demo.ipynb` 笔记本以获得更具交互性的体验。

## 工作原理

RiCA 通过为 LLM 引入“多线程思考”能力来工作。当 LLM 需要使用外部工具时，它可以选择在后台运行该工具。在工具运行期间，LLM 可以继续推理、处理信息，甚至调用其他工具。当后台工具完成时，其结果将无缝插入回 LLM 的上下文中，使其能够将新信息纳入其推理过程。

## 文档

- [开发指南](DEVELOPMENT.md)
- [贡献指南](CONTRIBUTING.md)
