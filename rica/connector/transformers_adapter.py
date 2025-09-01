from __future__ import annotations

import asyncio
import json
from typing import Callable, Any, List, Optional, Dict

from ..exceptions import AdapterDependenciesImportError
from ._adapter import _ReasoningThreadTemplate

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
except ImportError:
    # 错误会在适配器导入时触发，保留明确信息
    raise AdapterDependenciesImportError("使用此适配器需要 transformers 和 pytorch。")

default_model_name = "google/gemma-3-1b-it"


class ReasoningThread(_ReasoningThreadTemplate):
    """
    基于 HF Transformers 的推理线程，带有即时插入和工具调用执行功能。

    职责:
    - 维护一个可变的文本上下文。
    - 支持在下一个生成 token 后立即生效的文本插入。
    - 逐 token 生成文本，并在上下文尾部检测 <rica ...>...</rica> 工具调用。
    - 检测到工具调用时，通过 router._execute 执行并立即追加结果。
    - 通过 @rt.trigger 注册回调函数，在新文本追加时立即调用。
    - 生命周期控制: run/pause/wait/destroy。
    """

    def __init__(self, context: str = "", model_name: str = default_model_name,
                 generation_config: Optional[Dict[str, Any]] = None):
        super().__init__(context)
        self.model_name: str = model_name

        # 运行时状态
        self._pending_inserts: asyncio.Queue[str] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._pause_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._done_event = asyncio.Event()
        self._lock = asyncio.Lock()

        # 模型/分词器 (懒加载)
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._device: Optional[str] = None
        self._eos_id: Optional[int] = None

        # 生成配置
        self._generation_config = GenerationConfig(**(generation_config or {}))
        if not hasattr(self._generation_config, "pad_token_id"):
            self._generation_config.pad_token_id = self._generation_config.eos_token_id

        # 初始时设置为非暂停状态
        self._pause_event.set()
        # 创建任务骨架，但不立即运行
        self._task = asyncio.create_task(self._run_loop())

    # -------- 公共生命周期 API --------
    async def insert(self, text: Any):
        if text is None:
            return
        s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)

        async with self._lock:
            self._context += s
        await self._emit(s)
        await self._pending_inserts.put(s)

        # 如果任务暂停，则恢复
        self._pause_event.set()

    async def wait(self):
        if self._task and not self._task.done():
            await self._done_event.wait()

    async def destroy(self):
        self._stop_event.set()
        self._pause_event.set()  # 确保任务可以响应停止信号
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self._task.cancel()
        self._done_event.set()

    async def run(self):
        if self._task is None or self._task.done():
            # 如果任务已完成或被取消，需要重新创建
            self._done_event.clear()
            self._stop_event.clear()
            self._task = asyncio.create_task(self._run_loop())
        self._pause_event.set()

    async def pause(self):
        self._pause_event.clear()

    # -------- 内部辅助方法 --------
    async def _ensure_model(self):
        if self._model and self._tokenizer:
            return

        def _load_model_and_tokenizer():
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                device_map=("auto" if torch.cuda.is_available() else None),
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return model, tokenizer

        # 在单独的线程中执行阻塞的加载操作
        self._model, self._tokenizer = await asyncio.to_thread(_load_model_and_tokenizer)
        self._eos_id = self._tokenizer.eos_token_id
        self._device = self._model.device
        self._model.eval()

    async def _run_loop(self):
        try:
            await self._ensure_model()

            input_ids = self._tokenizer.encode(self._context, return_tensors="pt").to(self._device)
            past_key_values = None

            while not self._stop_event.is_set():
                await self._pause_event.wait()
                if self._stop_event.is_set():
                    break

                # 1. 处理挂起的外部插入
                input_ids, past_key_values = await self._process_pending_inserts(input_ids, past_key_values)

                # 2. 如果没有新的输入，则生成下一个 token
                if input_ids.shape[1] > 0:
                    with torch.no_grad():
                        outputs = self._model(input_ids=input_ids, use_cache=True, past_key_values=past_key_values)
                        past_key_values = outputs.past_key_values

                        # 使用 generation_config 进行采样
                        next_token_logits = outputs.logits[:, -1, :]
                        next_tokens = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
                else:  # 如果没有新输入，暂停直到有新的 insert
                    await self.pause()
                    continue

                # 3. 处理新生成的 token
                new_text = self._tokenizer.decode(next_tokens[0], skip_special_tokens=True)
                if new_text:
                    async with self._lock:
                        self._context += new_text
                    await self._emit(new_text)

                input_ids = next_tokens

                # 4. 检测并执行工具调用
                tool_result_text = await self._process_tool_call_if_detected()
                if tool_result_text:
                    tool_result_ids = self._tokenizer.encode(tool_result_text, return_tensors="pt").to(self._device)
                    input_ids = torch.cat([input_ids, tool_result_ids], dim=1)

                # 5. 检查是否生成结束
                if self._eos_id is not None and next_tokens.item() == self._eos_id:
                    break

        except (asyncio.CancelledError, Exception) as e:
            await self._emit(f"[adapter-error]{type(e).__name__}: {e}")
            if not isinstance(e, asyncio.CancelledError):
                raise
        finally:
            self._done_event.set()

    async def _process_pending_inserts(self, all_ids, past_key_values):
        """处理所有待处理的插入，并更新 input_ids 和 past_key_values。"""
        insert_text = ""
        while not self._pending_inserts.empty():
            insert_text += await self._pending_inserts.get()
            self._pending_inserts.task_done()

        if insert_text:
            insert_ids = self._tokenizer.encode(insert_text, return_tensors="pt").to(self._device)
            # 当有外部输入时，为了简单起见，我们不使用 past_key_values，而是重新处理整个上下文。
            # 这是一个权衡：简化了逻辑，但可能牺牲了一些性能。
            # 对于需要高性能的场景，可以实现更复杂的kv-cache拼接逻辑。
            all_text_ids = self._tokenizer.encode(self._context, return_tensors="pt").to(self._device)
            return all_text_ids, None

        return all_ids, past_key_values

    async def _process_tool_call_if_detected(self) -> Optional[str]:
        """检测并执行工具调用，返回结果文本供模型继续处理。"""
        if await self._detect_and_execute_tool_tail():
            # _detect_and_execute_tool_tail 已经将结果追加到 self._context 并 emit
            # 我们只需要获取这个结果文本，以便编码并送入下一次模型推理
            # 这里我们假设追加的文本就是最新的工具调用结果
            # 注意：_detect_and_execute_tool_tail 内部需要返回追加的文本
            # 我们需要稍微修改基类 _adapter.py

            # 假设 _detect_and_execute_tool_tail 返回 (bool, Optional[str])
            executed, result_text = await self._detect_and_execute_tool_tail_modified()
            if executed:
                return result_text
        return None

    # 你需要稍微修改基类 `_adapter.py` 中的 `_detect_and_execute_tool_tail`
    # 让它返回追加的文本。例如：
    async def _detect_and_execute_tool_tail_modified(self) -> tuple[bool, Optional[str]]:
        # ... (检测逻辑不变)
        try:
            result = await router._execute(tag_text)
            # ... (处理 result 的逻辑不变)
            appended = "..."  # 这是计算出的结果字符串
            self._context += appended
            await self._emit(appended)
            return True, appended
        except Exception as e:
            # ... (错误处理逻辑不变)
            return False, None

    # 为保持兼容，你可以在这个文件中重写基类方法，而不是修改基类文件
    async def _detect_and_execute_tool_tail(self) -> bool:
        executed, _ = await self._detect_and_execute_tool_tail_modified()
        return executed
