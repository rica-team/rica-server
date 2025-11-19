import pytest
import os
import tarfile
import tempfile
from rica.adapters.base import ReasoningThreadBase
from rica.core.application import RiCA

# Mock ReasoningThread for testing
class MockReasoningThread(ReasoningThreadBase):
    async def insert(self, text): pass
    async def wait(self): pass
    async def destroy(self): pass
    def run(self): pass
    def pause(self): pass

@pytest.mark.asyncio
async def test_install_from_py_file():
    thread = MockReasoningThread()
    
    # Create a dummy app file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        f.write("from rica.core.application import RiCA\n")
        f.write("app = RiCA('test.file_app')\n")
        f.write("@app.route('/hello', background=False)\n")
        f.write("def hello(data): return 'world'\n")
        app_path = f.name
        
    try:
        await thread.install(app_path)
        assert "test.file_app" in thread._apps
        
        # Verify app is loaded correctly
        app = thread._apps["test.file_app"]
        assert app.find_route("/hello") is not None
    finally:
        if os.path.exists(app_path):
            os.remove(app_path)

@pytest.mark.asyncio
async def test_install_from_tar_gz():
    thread = MockReasoningThread()
    
    # Create a dummy app structure
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_dir = os.path.join(tmpdir, "my_pkg")
        os.makedirs(pkg_dir)
        
        with open(os.path.join(pkg_dir, "__init__.py"), "w") as f:
            f.write("from rica.core.application import RiCA\n")
            f.write("app = RiCA('test.tar_app')\n")
        
        # Create tar.gz
        tar_path = os.path.join(tmpdir, "app.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(pkg_dir, arcname="my_pkg")
            
        await thread.install(tar_path)
        assert "test.tar_app" in thread._apps
