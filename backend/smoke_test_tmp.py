"""POST 计算端点冒烟: 测不依赖行情或本地计算路径, 网络类短超时快速判可用。"""
import warnings
warnings.filterwarnings("ignore")
import json

from fastapi.testclient import TestClient
from app.main import app

c = TestClient(app)


def call(method, path, **kw):
    r = getattr(c, method)(path, **kw)
    status = r.status_code
    body = r.text[:200]
    return status, body


# 登录拿 token
r = c.post("/api/auth/login", json={"username": "smoketest", "password": "pw123456"})
token = r.json().get("token", "") if r.status_code == 200 else ""
H = {"Authorization": f"Bearer {token}"} if token else {}

results = []


def record(name, status, body):
    ok = "PASS" if status < 400 else "FAIL"
    results.append((name, status, ok, body))
    print(f"[{ok}] {name} -> {status} | {body[:160]}")


print("=== 本地计算类(无网络) ===")

# 1. portfolio-opt 纯数值
record("/api/portfolio-opt (mean_variance)",
       *call("post", "/api/portfolio-opt", json={
           "codes": ["a", "b", "c"],
           "mu": [0.10, 0.15, 0.08],
           "cov": [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.05]],
           "method": "mean_variance",
           "maxWeight": 0.4,
       }))

record("/api/portfolio-opt (max_sharpe)",
       *call("post", "/api/portfolio-opt", json={
           "codes": ["a", "b"],
           "mu": [0.10, 0.15],
           "cov": [[0.04, 0.01], [0.01, 0.09]],
           "method": "max_sharpe",
       }))

record("/api/portfolio-opt (risk_parity)",
       *call("post", "/api/portfolio-opt", json={
           "codes": ["a", "b", "c"],
           "mu": [0.1, 0.1, 0.1],
           "cov": [[0.04, 0, 0], [0, 0.05, 0], [0, 0, 0.06]],
           "method": "risk_parity",
       }))

record("/api/portfolio-opt (equal)",
       *call("post", "/api/portfolio-opt", json={
           "codes": ["a", "b"],
           "mu": [0.1, 0.2],
           "cov": [[0.04, 0], [0, 0.09]],
           "method": "equal",
       }))

# 2. reports 导出(本地)
record("/api/reports/backtest (html)",
       *call("post", "/api/reports/backtest", json={
           "format": "html",
           "factorLabel": "动量",
           "config": {"factor": "momentum"},
           "metrics": {"cumulativeReturn": 0.12, "annualReturn": 0.08, "maxDrawdown": -0.05, "sharpe": 1.2},
           "groupSummary": [],
           "longShort": [],
           "icSeries": [],
       }))

record("/api/reports/backtest (excel)",
       *call("post", "/api/reports/backtest", json={
           "format": "excel",
           "factorLabel": "动量",
           "config": {"factor": "momentum"},
           "metrics": {"cumulativeReturn": 0.12},
       }))

# 3. portfolio/reset(写库, 登录)
record("/api/portfolio/reset", *call("post", "/api/portfolio/reset", headers=H))

# 4. strategies 保存
rstrat = call("post", "/api/strategies", json={
    "name": "冒烟策略",
    "kind": "backtest",
    "config": {"factor": "momentum"},
}, headers=H)
record("/api/strategies (save)", *rstrat)
# 提取 sid
sid = None
try:
    sid = rstrat[1] and json.loads(rstrat[1]).get("id")
except Exception:
    pass

# 5. user-factors 保存
ruserf = call("post", "/api/user-factors", json={
    "name": "冒烟组合因子",
    "kind": "composite",
    "definition": {"factors": [{"key": "momentum", "weight": 1.0}]},
}, headers=H)
record("/api/user-factors (save)", *ruserf)

# 6. backtest-runs 保存
rbtrun = call("post", "/api/backtest-runs", json={
    "strategyId": sid,
    "config": {"factor": "momentum"},
    "metrics": {"sharpe": 1.0},
}, headers=H)
record("/api/backtest-runs (save)", *rbtrun)

# 7. optimize/save-strategy(登录, 本地)
record("/api/optimize/save-strategy", *call("post", "/api/optimize/save-strategy", json={
    "name": "优化冒烟",
    "baseConfig": {"factor": "momentum"},
    "bestParams": {"n": 5},
}, headers=H))

# 8. risk 归因(GET, 但依赖持仓; 测空持仓路径)
record("/api/risk/attribution (空持仓)", *call("get", "/api/risk/attribution", headers=H))

print("\n=== 网络行情类(短超时判可用) ===")
# 这些依赖腾讯/新浪/东财接口; 测试环境若断网会 5xx, 联网则 200
# 用最小参数, poolSize 小, hist 短, 减少等待

# quote 实时行情
record("/api/quote (实时)", *call("get", "/api/quote", params={"codes": "sh600000,sz000001"}))

# select/backtest 小参数
record("/api/select/backtest (网络)", *call("post", "/api/select/backtest", json={
    "factor": "momentum", "poolSize": 20, "hist": 30, "n": 5, "groups": 5,
}))

# regression
record("/api/regression (网络)", *call("post", "/api/regression", json={
    "codes": ["sh600000"], "factor": "momentum", "hist": 30, "n": 5,
}))

# intraday
record("/api/intraday/backtest (网络)", *call("post", "/api/intraday/backtest", json={
    "code": "sh600000", "count": 48, "period": "5",
}))

# portfolio-opt/estimate
record("/api/portfolio-opt/estimate (网络)", *call("post", "/api/portfolio-opt/estimate", json={
    "codes": ["sh600000", "sz000001"], "hist": 30,
}))

print("\n=== 异步任务 jobs(提交后轮询) ===")
rjob = c.post("/api/jobs", json={"kind": "backtest", "config": {"factor": "momentum", "poolSize": 20, "hist": 30}})
job_id = None
if rjob.status_code == 200:
    try:
        job_id = rjob.json().get("jobId")
    except Exception:
        pass
record("/api/jobs (submit backtest)", rjob.status_code, rjob.text[:160])

if job_id:
    import time
    time.sleep(3)
    rpoll = c.get(f"/api/jobs/{job_id}")
    record(f"/api/jobs/{job_id} (poll)", *call("get", f"/api/jobs/{job_id}"))
    # 取消
    rcancel = c.delete(f"/api/jobs/{job_id}")
    record(f"/api/jobs/{job_id} (cancel)", rcancel.status_code, rcancel.text[:160])

print("\n=== 汇总 ===")
passed = sum(1 for _, _, ok, _ in results if ok == "PASS")
failed = len(results) - passed
print(f"PASS {passed} / FAIL {failed} / TOTAL {len(results)}")

# 写结果到 json 供报告使用
open("smoke_post_result.json", "w", encoding="utf-8").write(
    json.dumps([{"name": n, "status": s, "ok": o, "body": b} for n, s, o, b in results],
               ensure_ascii=False, indent=2)
)
print("\nDONE")
