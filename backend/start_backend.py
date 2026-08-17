import subprocess
import sys
import time
import urllib.request
import os

PORT = 8001
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_backend_8001.log")

# 关闭旧进程（按端口占用不可靠，这里用 pid 文件）
pid_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_backend.pid")
if os.path.exists(pid_file):
    try:
        old = int(open(pid_file).read().strip())
        import signal
        os.kill(old, signal.SIGTERM)
        time.sleep(1)
    except Exception:
        pass

logf = open(LOG, "w")
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
     "--port", str(PORT), "--log-level", "warning"],
    cwd=os.path.dirname(os.path.abspath(__file__)),
    stdout=logf, stderr=subprocess.STDOUT,
    creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) |
                 getattr(subprocess, "DETACHED_PROCESS", 0),
)
with open(pid_file, "w") as f:
    f.write(str(proc.pid))

# 等待就绪
for _ in range(30):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/health", timeout=2)
        print(f"BACKEND_READY pid={proc.pid} port={PORT}")
        sys.exit(0)
    except Exception:
        time.sleep(1)

print("BACKEND_START_FAILED")
print(open(LOG).read()[-2000:])
sys.exit(1)
