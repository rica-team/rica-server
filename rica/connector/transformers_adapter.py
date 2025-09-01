from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, List, Optional

from ..exceptions import AdapterDependenciesImportError
from ._adapter import _ReasoningThreadTemplate

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
except ImportError:
    # The error will be raised upon adapter import; the message is explicit.
    raise AdapterDependenciesImportError(
        "The transformers adapter requires 'transformers' and 'torch' to be installed.")

default_model_name = "google/gemma-3-1b-it"


class ReasoningThread(_ReasoningThreadTemplate):
    """
    A reasoning thread based on Hugging Face Transformers that supports a token-by-token
    generation loop with real-time text insertion and tool-call execution.

    Responsibilities:
    - Maintain a mutable string context for the language model.
    - Support immediate text insertions that are processed before the next generation step.
    - Generate tokens one by one, continuously checking the context tail for <rica> tool calls.
    - Execute detected tool calls via the router and append their results to the context.
    - Provide a decorator `@rt.trigger` to register callbacks for newly generated text.
    - Offer lifecycle controls: run, pause, wait, and destroy.
    """

    def __init__(
            self,
            context: str = "",
            model_name: str = default_model_name,
            generation_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(context)
        self.model_name: str = model_name

        # --- Runtime State ---
        self._pending_inserts: asyncio.Queue[str] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # --- Control Events ---
        self._pause_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._done_event = asyncio.Event()
        self._pause_event.set()  # Start in a running state

        # --- Model & Tokenizer (lazy-loaded) ---
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._device: Optional[str] = None

        # --- Generation Configuration ---
        # Set default generation strategy if not provided.
        default_config = {"use_cache": True, "do_sample": True, "temperature": 0.7, "top_p": 0.95}
        if generation_config:
            default_config.update(generation_config)
        self._generation_config = GenerationConfig(**default_config)

        # Start the main loop task upon initialization. It will wait for run() or insert().
        self._task = asyncio.create_task(self._run_loop())

    # --------------------------------------------------------------------------
    # Public Lifecycle API
    # --------------------------------------------------------------------------

    async def insert(self, text: Any):
        """
        Inserts external text into the context. The text will be processed
        before the next model generation step. This also resumes the thread if paused.
        """
        if text is None:
            return
        s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)

        # The context is updated immediately for external readers.
        async with self._lock:
            self._context += s
        await self._emit(s)
        await self._pending_inserts.put(s)

        # Ensure the generation loop is running to process the insert.
        self.run()

    async def wait(self):
        """Waits until the current generation task completes (e.g., hits EOS or is stopped)."""
        if self._task and not self._task.done():
            await self._done_event.wait()

    async def destroy(self):
        """Signals the generation task to stop and cleans up resources."""
        if self._task and not self._task.done():
            self._stop_event.set()
            self._pause_event.set()  # Unblock the loop if it's paused
            try:
                # Wait gracefully for the task to finish
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self._task.cancel()  # Forcefully cancel if it doesn't stop
        self._done_event.set()

    def run(self):
        """Starts or resumes the reasoning loop."""
        if self._task is None or self._task.done():
            # Recreate the task if it has been destroyed or finished
            self._done_event.clear()
            self._stop_event.clear()
            self._task = asyncio.create_task(self._run_loop())
        self._pause_event.set()

    def pause(self):
        """Pauses the reasoning loop after the current token is generated."""
        self._pause_event.clear()

    # --------------------------------------------------------------------------
    # Internal Helper Methods
    # --------------------------------------------------------------------------

    async def _ensure_model(self):
        """
        Loads the model and tokenizer on the first call, running the blocking
        I/O operations in a separate thread to avoid blocking the asyncio event loop.
        """
        if self._model and self._tokenizer:
            return

        def _load_from_pretrained():
            """Synchronous loading function to be run in a thread."""
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                device_map=("auto" if torch.cuda.is_available() else "cpu"),
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            self._generation_config.pad_token_id = tokenizer.pad_token_id
            self._generation_config.eos_token_id = tokenizer.eos_token_id
            return model, tokenizer

        self._model, self._tokenizer = await asyncio.to_thread(_load_from_pretrained)
        self._device = self._model.device
        self._model.eval()

    async def _run_loop(self):
        """The main generation loop."""
        try:
            await self._ensure_model()

            # Initialize with the current context.
            input_ids = self._tokenizer.encode(self._context, return_tensors="pt").to(self._device)
            past_key_values = None
            attention_mask = torch.ones_like(input_ids)

            while not self._stop_event.is_set():
                await self._pause_event.wait()
                if self._stop_event.is_set():
                    break

                # --- Step 1: Process any pending text insertions ---
                input_ids, past_key_values, attention_mask = await self._process_pending_inserts(
                    input_ids, past_key_values, attention_mask
                )

                if input_ids.shape[1] == 0:
                    # If there's nothing to process, pause until new input arrives.
                    self.pause()
                    continue

                # --- Step 2: Generate the next token ---
                with torch.no_grad():
                    outputs = self._model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    next_token_logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values

                    # Apply sampling strategy
                    if self._generation_config.do_sample:
                        probs = torch.softmax(next_token_logits / self._generation_config.temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # --- Step 3: Decode, append, and emit the new token ---
                new_text = self._tokenizer.decode(next_token[0], skip_special_tokens=True)
                if new_text:
                    async with self._lock:
                        self._context += new_text
                    await self._emit(new_text)

                # --- Step 4: Check for and execute tool calls ---
                tool_result_ids = await self._process_tool_call_if_detected()
                if tool_result_ids:
                    # If a tool was called, its result becomes the next input.
                    input_ids = torch.cat([next_token, tool_result_ids], dim=1)
                else:
                    # Otherwise, the generated token is the next input.
                    input_ids = next_token

                attention_mask = torch.ones_like(input_ids)

                # --- Step 5: Check for End-Of-Sequence token ---
                if self._generation_config.eos_token_id is not None and next_token.item() == self._generation_config.eos_token_id:
                    break

        except Exception as e:
            error_msg = f"[adapter-error]{type(e).__name__}: {e}"
            await self._emit(error_msg)
            if not isinstance(e, asyncio.CancelledError):
                # Re-raise unexpected errors
                raise
        finally:
            self._done_event.set()

    async def _process_pending_inserts(self, input_ids, past_key_values, attention_mask):
        """Drains the insert queue and prepares the new input tensors for the model."""
        if self._pending_inserts.empty():
            return input_ids, past_key_values, attention_mask

        insert_text = ""
        while not self._pending_inserts.empty():
            insert_text += self._pending_inserts.get_nowait()
            self._pending_inserts.task_done()

        # When external text is inserted, it's safest to invalidate the KV cache
        # and re-process the entire context to ensure correctness. This is a
        # trade-off between performance and logical simplicity.
        new_input_ids = self._tokenizer.encode(self._context, return_tensors="pt").to(self._device)
        new_attention_mask = torch.ones_like(new_input_ids)
        return new_input_ids, None, new_attention_mask

    async def _process_tool_call_if_detected(self) -> Optional[torch.Tensor]:
        """
        Checks for a tool call at the end of the context. If found, executes it
        and returns the encoded result as a tensor to be fed back to the model.
        """
        context_before = self.context
        was_executed = await super()._detect_and_execute_tool_tail()

        if was_executed:
            # The base method updated self.context and emitted the result.
            # We calculate the delta to feed it back into the model.
            newly_added_text = self.context[len(context_before):]
            if newly_added_text:
                return self._tokenizer.encode(newly_added_text, return_tensors="pt").to(self._device)
        return None