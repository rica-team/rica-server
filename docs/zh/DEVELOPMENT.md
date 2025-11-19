# 开发指南

## 安装

要设置开发环境，请运行：

```bash
pip install .[dev,pt]
```

## 运行测试

我们使用 `pytest` 进行测试。运行以下命令执行测试：

```bash
pytest
```

## 代码风格

我们使用 `ruff` 和 `black` 进行代码格式化和检查。

```bash
ruff check .
black .
```

## 项目结构

- `rica/`: 源代码
  - `core/`: 核心应用逻辑
  - `adapters/`: LLM 适配器（例如 Transformers）
  - `utils/`: 工具函数
- `tests/`: 单元测试和集成测试
- `demo/`: 示例脚本和笔记本
- `docs/`: 文档
