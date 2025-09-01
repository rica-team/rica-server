from __future__ import annotations

import asyncio
import json
from threading import Thread
from typing import Any, Dict, Optional

from ..exceptions import AdapterDependenciesImportError
from ..server import RiCA
from ._adapter import _ReasoningThreadTemplate

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        GenerationConfig,
        TextIteratorStreamer,
        StoppingCriteria,
        StoppingCriteriaList,
    )
except ImportError:
    raise AdapterDependenciesImportError(
        "The transformers adapter requires 'transformers' and 'torch' to be installed."
    )

default_model_name = "google/gemma-2-2b-it"


class _ToolCallStoppingCriteria(StoppingCriteria):
    """Stops generation when a complete <rica>...</rica> tag is detected."""

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Decode the entire generated sequence
        full_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # Check if a complete tool call tag exists at the end of the text
        if full_text.rstrip().endswith("</rica>"):
            return True
        return False


class ReasoningThread(_ReasoningThreadTemplate):
    """
    A reasoning thread based on Hugging Face Transformers that supports a token-by-token
    generation loop with real-time text insertion and tool-call execution.

    This implementation leverages the `model.generate` method with a `TextIteratorStreamer`
    for efficient, non-blocking token generation, and a custom `StoppingCriteria` to
    interrupt generation upon detecting a complete tool call.
    """

    def __init__(
            self,
            app: RiCA,
            context: str = "",
            model_name: str = default_model_name,
            generation_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(app, context)
        self.model_name: str = model_name

        # --- Runtime State ---
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # --- Control Events ---
        self._pause_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._is_running = False
        self._pause_event.set()  # Start in a running state

        # --- Model & Tokenizer (lazy-loaded) ---
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._streamer: Optional[TextIteratorStreamer] = None
        self._stopping_criteria: Optional[StoppingCriteriaList] = None

        # --- Generation Configuration ---
        default_config = {"max_new_tokens": 1024, "do_sample": True, "temperature": 0.6, "top_p": 0.9}
        if generation_config:
            default_config.update(generation_config)
        self._generation_config = GenerationConfig(**default_config)

    # --------------------------------------------------------------------------
    # Public Lifecycle API
    # --------------------------------------------------------------------------

    async def insert(self, text: Any):
        """
        Inserts external text into the context. This will pause any ongoing generation,
        append the text, and then resume generation from the new context.
        """
        if text is None:
            return
        s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)

        # Pause generation to safely modify the context
        self.pause()
        async with self._lock:
            self._context += s
        await self._emit(s)

        # Resume generation, which will now use the updated context
        self.run()

    async def wait(self):
        """Waits until the main reasoning task is complete or stopped."""
        if self._task and not self._task.done():
            await self._task

    async def destroy(self):
        """Signals the generation task to stop and cleans up resources."""
        if self._task and not self._task.done():
            self._stop_event.set()
            self._pause_event.set()  # Unblock if paused
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self._task.cancel()

    def run(self):
        """Starts or resumes the reasoning loop."""
        self._pause_event.set()
        if not self._is_running:
            self._is_running = True
            self._stop_event.clear()
            self._task = asyncio.create_task(self._run_loop())

    def pause(self):
        """Pauses the reasoning loop before the next generation cycle."""
        self._pause_event.clear()

    # --------------------------------------------------------------------------
    # Internal Helper Methods
    # --------------------------------------------------------------------------

    async def _ensure_model(self):
        """Loads the model and tokenizer on the first call in a separate thread."""
        if self._model and self._tokenizer:
            return

        def _load_sync():
            """Synchronous loading function to be run in a thread."""
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
        self._streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        self._stopping_criteria = StoppingCriteriaList([_ToolCallStoppingCriteria(self._tokenizer)])
        self._model.eval()

    async def _run_loop(self):
        """The main generation loop."""
        try:
            await self._ensure_model()

            while not self._stop_event.is_set():
                await self._pause_event.wait()
                if self._stop_event.is_set():
                    break

                # --- Step 1: Prepare inputs for generation ---
                async with self._lock:
                    inputs = self._tokenizer(self._context, return_tensors="pt").to(self._model.device)

                # --- Step 2: Start generation in a separate thread ---
                generation_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "generation_config": self._generation_config,
                    "streamer": self._streamer,
                    "stopping_criteria": self._stopping_criteria,
                }
                thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
                thread.start()

                # --- Step 3: Stream generated tokens and update context ---
                async for new_text in self._streamer:
                    if self._stop_event.is_set() or not self._pause_event.is_set():
                        # If stopped or paused externally, break the streaming loop
                        # Note: The underlying generate call will continue until it finishes.
                        # A more advanced implementation might need a way to signal it to stop.
                        break

                    async with self._lock:
                        self._context += new_text
                    await self._emit(new_text)

                # Wait for the generation thread to finish
                await asyncio.to_thread(thread.join)

                # --- Step 4: Check for and execute tool calls ---
                was_executed, _ = await self._detect_and_execute_tool_tail()
                if not was_executed:
                    # If no tool was called, it means generation ended naturally (EOS or max_tokens).
                    # We can pause here until new input arrives.
                    self.pause()

        except Exception as e:
            error_msg = f"[adapter-error]{type(e).__name__}: {e}"
            await self._emit(error_msg)
            if not isinstance(e, asyncio.CancelledError):
                raise
        finally:
            self._is_running = False
