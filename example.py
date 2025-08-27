import router
from server import RiCA

import subprocess
import time

app = RiCA()

@app.new("sys.sh.exec", background=False, timeout=10000)
def _sys_sh_exec(input_, *args, **kwargs):
    """
    A tool to execute shell commands
    input format:
        {
            "cmd": "ls -l",
            "timeout": 10000 # optional, in ms
        }
    output format:
        {
            "output": "...",
            "return_code": 0,
            "during": 300 # in ms
        }
    """
    # Validate input
    if not isinstance(input_, dict) or "cmd" not in input_ or not isinstance(input_["cmd"], str) or not input_["cmd"].strip():
        return {
            "output": "Invalid input: 'cmd' is required and must be a non-empty string.",
            "return_code": 1,
            "during": 0
        }

    cmd = input_["cmd"].strip()

    timeout_ms = input_.get("timeout", None)
    timeout_sec = None
    if isinstance(timeout_ms, (int, float)) and timeout_ms > 0:
        timeout_sec = timeout_ms / 1000.0

    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_sec
        )
        end = time.perf_counter()
        during_ms = int((end - start) * 1000)

        return {
            "output": (result.stdout or "") + (result.stderr or ""),
            "return_code": result.returncode,
            "during": during_ms
        }
    except subprocess.TimeoutExpired as e:
        end = time.perf_counter()
        during_ms = int((end - start) * 1000)
        out = (e.stdout or "") + (e.stderr or "")
        return {
            "output": out if out else f"Process timed out after {int(timeout_sec * 1000)} ms.",
            "return_code": -1,
            "during": during_ms
        }
    except Exception as e:
        end = time.perf_counter()
        during_ms = int((end - start) * 1000)
        return {
            "output": f"Execution failed: {type(e).__name__}: {e}",
            "return_code": -1,
            "during": during_ms
        }

if __name__ == "__main__":
    router.preset(
        application=app
    )
    while True:
        router._execute(input_=input(">>> "))

