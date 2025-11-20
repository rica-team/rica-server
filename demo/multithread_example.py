import asyncio
import logging

from rica.adapters.transformers_adapter import ReasoningThread
from rica.core.application import RiCA
from rica.core.thread_manager import get_thread_manager
from rica.core.whiteboard import get_whiteboard

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    # Initialize apps
    whiteboard_app = RiCA("rica")
    wb = get_whiteboard()

    @whiteboard_app.route("/whiteboard", background=False)
    def whiteboard_handler(data):
        return wb.handle(data)

    thread_manager_app = RiCA("rica.thread")
    tm = get_thread_manager()

    @thread_manager_app.route("/spawn", background=False)
    async def spawn_handler(data):
        # In a real scenario, 'parent_thread' would be the calling thread.
        # Here we mock it or pass the main thread if we can access it.
        # For this demo, we'll assume the main thread is the parent.
        return await tm.spawn(main_thread, data)

    @thread_manager_app.route("/kill", background=False)
    async def kill_handler(data):
        return await tm.kill(data)

    @thread_manager_app.route("/list", background=False)
    def list_handler(data):
        return tm.list(data)

    # Create main thread
    # Note: You need a real model name here or set RICA_DEFAULT_MODEL env var
    main_thread = ReasoningThread(model_name="Qwen/Qwen2.5-1.5B-Instruct")

    # Install apps
    await main_thread.install(whiteboard_app)
    await main_thread.install(thread_manager_app)

    await main_thread.initialize()

    print("Main thread initialized. Starting interaction...")

    # Simulate a complex task
    prompt = (
        "I want you to solve a complex problem using multi-threading.\n"
        "1. Spawn a sub-thread to calculate 10 + 10 and write the result to whiteboard "
        "'math_result' with description 'Simple Math'.\n"
        "2. Wait for a bit (simulated by you just waiting).\n"
        "3. Read the whiteboard 'math_result' and tell me the answer.\n"
        'Use <rica package="rica.thread" route="/spawn">...</rica> to spawn.\n'
        'Use <rica package="rica" route="/whiteboard">...</rica> to access whiteboard.'
    )

    # We need to mock the sub-thread's behavior because we don't have a real LLM running
    # that can follow instructions perfectly in this demo environment without a GPU/Model.
    # However, the `ReasoningThread` will try to load the model.
    # If the user doesn't have the model, this will fail.
    # So we should probably use a MockReasoningThread for the demo if we just want to
    # show the mechanism, OR we assume the user has the environment set up.
    # Given the user's context ("RiCA-Server"), they likely have the environment.
    # But to be safe and show the *mechanism*, maybe we can inject a "fake" model response?
    # The `ReasoningThread` uses `transformers`.

    # Let's just run it and see. If it fails, the user will know they need a model.
    # But wait, `spawn` creates a `ReasoningThread`.
    # If we want to demonstrate *without* a real model, we'd need to mock `ReasoningThread`.

    # For this file, let's write it as if it's a real usage example.

    await main_thread.insert(prompt)

    # Keep running for a bit
    await asyncio.sleep(10)

    await main_thread.destroy()


if __name__ == "__main__":
    asyncio.run(main())
