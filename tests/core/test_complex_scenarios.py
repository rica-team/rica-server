import pytest

from rica.adapters.base import ReasoningThreadBase
from rica.core.application import RiCA
from rica.core.thread_manager import get_thread_manager
from rica.core.whiteboard import get_whiteboard


class MockReasoningThread(ReasoningThreadBase):
    def __init__(self, context=""):
        super().__init__(context)
        self.generated_tokens = []
        self.sub_threads = []

    async def _emit_token(self, token: str):
        self.generated_tokens.append(token)

    async def _emit_response(self, response: dict):
        pass

    async def create_sub_thread(self, model_name=None, config=None):
        thread = MockReasoningThread()
        self.sub_threads.append(thread)
        return thread

    def run(self):
        pass

    async def initialize(self):
        pass

    async def destroy(self):
        pass

    async def insert(self, text):
        pass


@pytest.mark.asyncio
async def test_parallel_tool_execution():
    thread = MockReasoningThread()
    await thread.initialize()

    app = RiCA("test.pkg")

    @app.route("/tool_a", background=False)
    async def tool_a(data):
        return {"result": "A"}

    @app.route("/tool_b", background=False)
    async def tool_b(data):
        return {"result": "B"}

    await thread.install(app)

    # Simulate model outputting two tool calls
    thread._context = (
        '<rica package="test.pkg" route="/tool_a">{}</rica>'
        '<rica package="test.pkg" route="/tool_b">{}</rica>'
    )

    executed, result = await thread._detect_and_execute_tool_tail()

    assert executed
    assert '{"result": "A"}' in result
    assert '{"result": "B"}' in result
    assert result.count('{"result"') == 2


@pytest.mark.asyncio
async def test_whiteboard_interaction():
    wb = get_whiteboard()
    wb.handle({"action": "clear", "whiteboard_id": "test_wb"})

    # Write
    wb.handle(
        {
            "action": "write",
            "whiteboard_id": "test_wb",
            "content": "Shared Data",
            "description": "Test Data",
        }
    )

    # Read
    res = wb.handle({"action": "read", "whiteboard_id": "test_wb"})
    assert res["content"] == "Shared Data"
    assert res["description"] == "Test Data"

    # Append
    wb.handle({"action": "append", "whiteboard_id": "test_wb", "content": "More Data"})

    res = wb.handle({"action": "read", "whiteboard_id": "test_wb"})
    assert "Shared Data\nMore Data" == res["content"]

    # List
    res = wb.handle({"action": "list"})
    assert len(res["whiteboards"]) >= 1
    assert any(w["id"] == "test_wb" and w["description"] == "Test Data" for w in res["whiteboards"])


@pytest.mark.asyncio
async def test_thread_management():
    tm = get_thread_manager()
    parent = MockReasoningThread()

    # Spawn
    res = await tm.spawn(parent, {"task": "Sub-task"})
    assert res["status"] == "success"
    thread_id = res["thread_id"]

    assert tm.get_thread(thread_id) is not None
    assert len(parent.sub_threads) == 1

    # List
    threads = tm.list({})
    assert thread_id in threads["threads"]

    # Kill
    res = await tm.kill({"thread_id": thread_id})
    assert res["status"] == "success"
    assert tm.get_thread(thread_id) is None
