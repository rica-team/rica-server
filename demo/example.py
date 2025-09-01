import asyncio
from rica.server import RiCA
from rica.connector import transformers_adapter as tf

# 1. Create the RiCA application instance first.
app = RiCA()


# 2. Register tools with the application instance.
@app.register("sys.python.exec", background=False, timeout=5000)
async def _sys_python_exec(input_):
    """
    A tool to execute a single line of Python code.
    input:
    {
        "code": "1+1"
    }
    output:
    {
        "result": "2"
    }
    """
    try:
        code = input_.get("code", "")
        # Using eval is unsafe. For a real application, use a safer execution environment.
        result = eval(code)
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}


# 3. Create the ReasoningThread, passing the app instance to it.
# The thread now knows which tools it can use.
rt = tf.ReasoningThread(app, model_name="google/gemma-3-1b-it")


# 4. Register a trigger to handle output from the model.
@rt.trigger
def _output(content):
    # Print content token by token as it's generated.
    print(content, end="", flush=True)


async def main():
    print("--- Starting RiCA Demo ---")

    # 5. Start the reasoning thread. It will initialize the model and wait for input.
    rt.run()

    # 6. Insert the initial user prompt.
    await rt.insert(
        "User: Please calculate 123*456 using the Python execution tool. "
        "Think step-by-step about what you need to do."
    )

    # 7. Wait for the generation to complete (e.g., hit EOS or pause after tool call).
    # In a real app, you might have a more complex condition to wait for.
    await rt.wait()

    print("\n\n--- Final Context ---")
    print(rt.context)
    print("---------------------")

    # 8. Clean up resources.
    await rt.destroy()


if __name__ == "__main__":
    # The main entry point needs to be async.
    asyncio.run(main())
