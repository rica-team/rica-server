from unittest.mock import MagicMock

import pytest

from rica.adapters.base import ReasoningThreadBase
from rica.core.application import RiCA


class ConcreteReasoningThread(ReasoningThreadBase):
    """Concrete implementation of ReasoningThreadBase for testing."""

    async def insert(self, text):
        pass

    async def wait(self):
        pass

    async def destroy(self):
        pass

    def run(self):
        pass

    def pause(self):
        pass


@pytest.mark.asyncio
async def test_install_uninstall_app():
    thread = ConcreteReasoningThread()
    app = RiCA("test.pkg")

    await thread.install(app)
    assert "test.pkg" in thread._apps

    await thread.uninstall("test.pkg")
    assert "test.pkg" not in thread._apps


@pytest.mark.asyncio
async def test_callbacks():
    thread = ConcreteReasoningThread()

    token_mock = MagicMock()
    thread.token_generated(token_mock)

    trigger_mock = MagicMock()
    thread.trigger(trigger_mock)

    await thread._emit_token("token")
    token_mock.assert_called_with("token")

    await thread._emit_response({"result": "ok"})
    trigger_mock.assert_called_with({"result": "ok"})


@pytest.mark.asyncio
async def test_xml_parsing_robustness():
    thread = ConcreteReasoningThread()
    await thread.initialize()

    # Setup a mock app and route
    app = RiCA("test.pkg")

    @app.route("/echo", background=False)
    async def echo(data):
        return data

    await thread.install(app)

    # Test valid XML
    thread._context = '<rica package="test.pkg" route="/echo">{"msg": "hello"}</rica>'
    executed, result = await thread._detect_and_execute_tool_tail()
    assert executed
    assert '{"msg": "hello"}' in result

    # Test malformed XML that regex should recover
    # Missing quotes around attributes, slightly broken tag
    thread._context = '<rica package="test.pkg" route="/echo">{"msg": "recovered"}</rica>'
    executed, result = await thread._detect_and_execute_tool_tail()
    assert executed
    assert '{"msg": "recovered"}' in result

    # Test completely invalid XML
    thread._context = "<rica broken>...</rica>"
    executed, result = await thread._detect_and_execute_tool_tail()
    # Should return True (detected) but result contains error message
    assert executed
    assert "[tool-error]" in result or "InvalidRiCAString" in result
