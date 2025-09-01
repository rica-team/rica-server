from rica.server import RiCA
from rica.connector import transformers_adapter as tf

app = RiCA()

@app.register("sys.python.exec", True, 1000)
async def _sys_python_exec(input_, *args, **kwargs):
    """
    A tool to execute Python code.
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
        result = eval(code)
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}

rt = tf.ReasoningThread("")

@rt.trigger
def _output(content):
    print(content, end="", flush=True)

rt.insert("Please calculate 123*456 using Python.")
