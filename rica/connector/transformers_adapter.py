from __future__ import annotations

import asyncio
import json
from threading import Thread
from typing import Any, Dict, Optional

from ..exceptions import AdapterDependenciesImportError
from ..utils.prompt import _rica_prompt
from ._adapter import ReasoningThread

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        GenerationConfig,
        StoppingCriteria,
        StoppingCriteriaList,
        TextIteratorStreamer,
    )
except ImportError:
    raise AdapterDependenciesImportError(
        "The transformers adapter requires 'transformers' and 'torch' to be installed."
    )

default_model_name = "google/gemma-3-1b-it"


class _ToolCallStoppingCriteria(StoppingCriteria):
    """Stops generation when a complete <rica>...</rica> tag is detected."""

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        full_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if full_text.rstrip().endswith("</rica>"):
            return True
        return False


class TransformersReasoningThread(ReasoningThread):
    """
    A reasoning thread based on Hugging Face Transformers that supports a token-by-token
    generation loop with real-time text insertion and tool-call execution.
    """

    def __init__(
        self,
        context: str = "",
        model_name: str = default_model_name,
        generation_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(context)
        self.model_name: str = model_name
        self.model_modal: str = "PyTorch/Transformers"

        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._prompt_injected = False

        self._pause_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._is_running = False
        self._pause_event.set()

        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._streamer: Optional[TextIteratorStreamer] = None
        self._stopping_criteria: Optional[StoppingCriteriaList] = None

        default_config = {
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.9,
        }
        if generation_config:
            default_config.update(generation_config)
        self._generation_config = GenerationConfig(**default_config)

    # --------------------------------------------------------------------------
    # Public Lifecycle API
    # --------------------------------------------------------------------------

    async def insert(self, text: Any):
        if text is None:
            return

        s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
        formatted_input = f'<rica-callback package="rica.userinput">{s}</rica-callback>'

        self.pause()
        async with self._lock:
            self._context += formatted_input
        await self._emit_token(formatted_input)

        self.run()

    async def wait(self):
        if self._task and not self._task.done():
            await self._task

    async def destroy(self):
        if self._task and not self._task.done():
            self._stop_event.set()
            self._pause_event.set()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self._task.cancel()

    def run(self):
        self._pause_event.set()
        if not self._is_running:
            self._is_running = True
            self._stop_event.clear()
            self._task = asyncio.create_task(self._run_loop())

    def pause(self):
        self._pause_event.clear()

    # --------------------------------------------------------------------------
    # Internal Helper Methods
    # --------------------------------------------------------------------------

    def _get_next_token_from_streamer(self) -> Optional[str]:
        """
        Synchronous helper to safely get the next token from the streamer.
        It catches StopIteration and returns None, preventing the exception
        from propagating into the asyncio event loop.
        """
        try:
            return next(self._streamer)
        except StopIteration:
            return None

    async def _ensure_model(self):
        if self._model and self._tokenizer:
            return

        def _load_sync():
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return model, tokenizer

        self._model, self._tokenizer = await asyncio.to_thread(_load_sync)
        self._streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        self._stopping_criteria = StoppingCriteriaList([_ToolCallStoppingCriteria(self._tokenizer)])
        self._model.eval()

    async def _inject_prompt_if_needed(self):
        if self._prompt_injected:
            return

        async with self._lock:
            # The prompt now needs access to all installed apps
            async with self._apps_lock:
                system_prompt = await _rica_prompt(self._apps, self.model_name, self.model_modal)
            self._context = system_prompt + self._context
            self._prompt_injected = True
            print("--- System Prompt Injected ---")

    async def _run_loop(self):
        try:
            await self._ensure_model()

            while not self._stop_event.is_set():
                await self._pause_event.wait()
                if self._stop_event.is_set():
                    break

                await self._inject_prompt_if_needed()

                async with self._lock:
                    inputs = self._tokenizer(self._context, return_tensors="pt").to(
                        self._model.device
                    )

                generation_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "generation_config": self._generation_config,
                    "streamer": self._streamer,
                    "stopping_criteria": self._stopping_criteria,
                }
                thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
                thread.start()

                while True:
                    # Use the safe wrapper function with to_thread
                    new_text = await asyncio.to_thread(self._get_next_token_from_streamer)

                    if new_text is None:
                        # None indicates StopIteration was caught, so generation is finished.
                        break

                    if self._stop_event.is_set() or not self._pause_event.is_set():
                        break

                    async with self._lock:
                        self._context += new_text
                    await self._emit_token(new_text)

                await asyncio.to_thread(thread.join)

                was_executed, _ = await self._detect_and_execute_tool_tail()
                if not was_executed:
                    self.pause()

        except Exception as e:
            error_msg = f"[adapter-error]{type(e).__name__}: {e}"
            await self._emit_token(error_msg)
            if not isinstance(e, asyncio.CancelledError):
                raise
        finally:
            self._is_running = False
