from __future__ import annotations

import asyncio
import json
from typing import Callable, Any, List, Optional

from ..exceptions import AdapterDependenciesImportError
from ._adapter import _ReasoningThreadTemplate

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    # Import error will be raised on adapter import; keep explicit message.
    raise AdapterDependenciesImportError("transformers and pytorch are required to use this adapter.")


default_model_name = "google/gemma-3-1b-it"


class ReasoningThread(_ReasoningThreadTemplate):
    """
    HF Transformers-based Reasoning Thread with cut-in token loop and tool-call execution.

    Responsibilities:
    - Maintain a mutable context string.
    - Support immediate inserts that take effect right after the next generated token.
    - Generate tokens step-by-step, checking the tail of context for <rica ...>...</rica> tool strings.
    - When a tool string is detected, execute it via router._execute and append the result immediately.
    - Allow registering callbacks via @rt.trigger that are invoked immediately with newly appended text.
    - Lifecycle controls: run/pause/wait/destroy.
    """

    def __init__(self, context: str = "", model_name: str = default_model_name):
        super().__init__(context)
        self.model_name: str = model_name

        # Runtime state
        self._context: str = context or ""
        self._callbacks: List[Callable[[str], Any]] = []
        self._pending_inserts: asyncio.Queue[str] = asyncio.Queue()

        self._task: Optional[asyncio.Task] = None
        self._pause_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._done_event = asyncio.Event()
        self._pause_event.set()  # not paused initially

        self._lock = asyncio.Lock()

        # Model/tokenizer (lazy)
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._device = None
        self._eos_id: Optional[int] = None

    # -------- Public lifecycle API --------
    async def insert(self, text: Any):
        if text is None:
            return
        s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
        # Immediate append to context, emit, and queue for next-step feeding.
        async with self._lock:
            self._context += s
        await self._emit(s)
        self._pending_inserts.put_nowait(s)

        # Ensure generation is running
        if self._task is None or self._task.done():
            self._done_event.clear()
            self._stop_event.clear()
            self._task = asyncio.create_task(self._run_loop())
        else:
            # If paused, resume
            self._pause_event.set()

    async def wait(self):
        # Wait until current generation task completes (EOS or stop)
        if self._task is None:
            return
        await self._done_event.wait()

    async def destroy(self):
        # Signal stop and cancel task
        self._stop_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._done_event.set()

    async def run(self):
        # Start or resume
        if self._task is None or self._task.done():
            self._done_event.clear()
            self._stop_event.clear()
            self._task = asyncio.create_task(self._run_loop())
        self._pause_event.set()

    async def pause(self):
        self._pause_event.clear()

    @property
    def context(self) -> str:
        # Property is synchronous for easy access at any time
        return self._context

    # -------- Callback and base helpers are inherited from _ReasoningThreadTemplate --------

    async def _ensure_model(self):
        if self._model is not None and self._tokenizer is not None:
            return
        # Load lazily
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            device_map=("auto" if torch.cuda.is_available() else None),
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._eos_id = self._tokenizer.eos_token_id
        # Determine device
        self._device = self._model.device
        self._model.eval()

    async def _run_loop(self):
        try:
            await self._ensure_model()
            tokenizer = self._tokenizer
            model = self._model
            device = self._device
            eos_id = self._eos_id

            # Initialize input ids from current context
            all_ids = tokenizer.encode(self._context, return_tensors="pt").to(device)
            past_key_values = None
            current_input_ids = all_ids

            while not self._stop_event.is_set():
                # Pause handling
                await self._pause_event.wait()

                # Forward pass for current input ids
                with torch.no_grad():
                    outputs = model(input_ids=current_input_ids, use_cache=True, past_key_values=past_key_values)
                    logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values
                    next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

                # Append generated token
                all_ids = torch.cat([all_ids, next_token_id], dim=1)
                new_text = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
                if new_text:
                    async with self._lock:
                        self._context += new_text
                    await self._emit(new_text)

                # After each token, check for pending inserts
                pending_text = None
                try:
                    while True:
                        pending_text = self._pending_inserts.get_nowait()
                        if pending_text:
                            # append to input stream (already added to context above)
                            insert_ids = tokenizer.encode(pending_text, return_tensors="pt").to(device)
                            all_ids = torch.cat([all_ids, insert_ids], dim=1)
                            current_input_ids = insert_ids
                        self._pending_inserts.task_done()
                except asyncio.QueueEmpty:
                    pass

                # If there were no new inserts, next step feeds the just-generated token
                if current_input_ids is all_ids:
                    current_input_ids = next_token_id

                # Detect tool call at the tail
                if await self._detect_and_execute_tool_tail():
                    # After execution, feed the newly appended text next
                    # Take the most recent tail we just added (rough approach: encode last ~200 chars)
                    tail = self._context[-200:]
                    insert_ids = tokenizer.encode(tail, return_tensors="pt").to(device)
                    all_ids = torch.cat([all_ids, insert_ids], dim=1)
                    current_input_ids = insert_ids

                # Stop on EOS
                if eos_id is not None and int(next_token_id.item()) == int(eos_id):
                    break

            self._done_event.set()
        except asyncio.CancelledError:
            self._done_event.set()
            raise
        except Exception as e:
            # On error, mark done to unblock waiters
            self._done_event.set()
            # Also emit error text to callbacks for visibility
            await self._emit(f"[adapter-error]{type(e).__name__}: {e}")
        finally:
            # No special cleanup required
            ...





