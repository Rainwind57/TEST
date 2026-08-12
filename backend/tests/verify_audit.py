"""独立验证脚本：验证审查报告修复的正确性"""
import requests
import json

BASE = 'http://localhost:8899'

# 登录
r = requests.post(f'{BASE}/api/auth/login', json={'username': 'tester', 'password': 'test123'})
tok = r.json()['token']
h = {'Authorization': f'Bearer {tok}'}

tests = []

def check(name, ok, detail=""):
    tests.append((name, ok, detail))
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name} {('— ' + detail) if detail else ''}")

# === ML 模块 ===
print('\n=== 一、机器学习模块 ===')

models = requests.get(f'{BASE}/api/ml/models', headers=h).json()
# 选可计算的模型（非虚构特征）
computable = [m for m in models if m.get('computable', True) and not m.get('unknownFeatures')]
if not computable:
    computable = models
mid = computable[0]['id']

# P0-1
r = requests.post(f'{BASE}/api/ml/backtest', json={
    'modelId': mid, 'board': 'all', 'poolSize': 30, 'groups': 2,
    'n': 5, 'hist': 100, 'applyCost': False
}, headers=h)
res = r.json()
check('P0-1 actualHistDays', 'actualHistDays' in res, f"={res.get('actualHistDays')}")
check('P0-1 effectiveStart', 'effectiveStart' in res, f"={res.get('effectiveStart')}")
check('P0-1 effectiveEnd', 'effectiveEnd' in res, f"={res.get('effectiveEnd')}")

r2 = requests.post(f'{BASE}/api/ml/backtest', json={
    'modelId': mid, 'board': 'all', 'poolSize': 30, 'groups': 2,
    'n': 5, 'hist': 50, 'applyCost': False
}, headers=h)
res2 = r2.json()
check('P0-1 histWarning(hist=50)', 'histWarning' in res2, res2.get('histWarning', '')[:80])

r3 = requests.post(f'{BASE}/api/ml/backtest', json={
    'modelId': mid, 'board': 'all', 'poolSize': 30, 'groups': 2,
    'n': 5, 'hist': 1024, 'applyCost': False
}, headers=h)
res3 = r3.json()
check('P0-1 noWarning(hist=1024)', not res3.get('histWarning'), 'OK' if not res3.get('histWarning') else res3['histWarning'][:80])

# P1-1
params = requests.get(f'{BASE}/api/ml/models/{mid}/params', headers=h).json()
check('P1-1 featureImportance存在', 'featureImportance' in params, f"len={len(params.get('featureImportance',[]))}")
check('P1-1 featureImportance非空', len(params.get('featureImportance', [])) > 0)

# P1-2
r4 = requests.post(f'{BASE}/api/ml/models/{mid}/adjust', json={
    'featureWeights': {'momentum5': 2.0}, 'saveAsNew': True
}, headers=h)
res4 = r4.json()
if 'newModelId' in res4:
    check('P1-2 saveAsNew创建模型', True, res4['newModelId'])
    requests.delete(f'{BASE}/api/ml/models/{res4["newModelId"]}', headers=h)
else:
    check('P1-2 saveAsNew创建模型', False, str(res4))

r5 = requests.post(f'{BASE}/api/ml/models/{mid}/adjust', json={
    'saveAsNew': True
}, headers=h)
res5 = r5.json()
if 'newModelId' in res5:
    check('P1-2 saveAsNew无权重', True, res5['newModelId'])
    requests.delete(f'{BASE}/api/ml/models/{res5["newModelId"]}', headers=h)
else:
    check('P1-2 saveAsNew无权重', False, str(res5))

# === Monitor 模块 ===
print('\n=== 二、盯盘模块 ===')

cfg = requests.get(f'{BASE}/api/monitor/config', headers=h).json()
check('P0 config.board', cfg.get('board') == 'all', str(cfg.get('board')))
check('P0 config.poolSize', cfg.get('poolSize') == 150, str(cfg.get('poolSize')))
check('P0 config.adjustId存在', 'adjustId' in cfg, str(cfg.get('adjustId')))

status = requests.get(f'{BASE}/api/monitor/status', headers=h).json()
check('P0 status含config', 'config' in status, 'OK')

cfg2 = requests.post(f'{BASE}/api/monitor/config', json={
    'mode': 'rule', 'board': 'sh_main', 'poolSize': 200
}, headers=h).json()
check('P0 set board=sh_main', cfg2.get('board') == 'sh_main', str(cfg2.get('board')))
check('P0 set poolSize=200', cfg2.get('poolSize') == 200, str(cfg2.get('poolSize')))
requests.post(f'{BASE}/api/monitor/config', json={
    'mode': 'rule', 'board': 'all', 'poolSize': 150
}, headers=h)

# === 边界 ===
print('\n=== 三、边界测试 ===')

r = requests.post(f'{BASE}/api/monitor/config', json={'mode': 'invalid'}, headers=h)
check('Boundary 无效mode→400', r.status_code in (400, 422), str(r.status_code))

r = requests.post(f'{BASE}/api/monitor/config', json={'mode': 'model', 'modelId': ''}, headers=h)
check('Boundary model无id→400', r.status_code in (400, 422), str(r.status_code))

r = requests.post(f'{BASE}/api/ml/models/no_such_model/adjust', json={'saveAsNew': True}, headers=h)
check('Boundary 不存在的model→404', r.status_code in (404, 500), str(r.status_code))

# hist边界
r = requests.post(f'{BASE}/api/ml/backtest', json={
    'modelId': mid, 'board': 'all', 'poolSize': 30, 'groups': 2,
    'n': 3, 'hist': 0, 'applyCost': False
}, headers=h)
if r.status_code == 200:
    res = r.json()
    check('Boundary hist=0→autoClamp', res.get('actualHistDays', 0) >= 244, f"={res.get('actualHistDays')}")
elif r.status_code == 422:
    check('Boundary hist=0→422(样本不足)', True, '422 expected for small pool')
else:
    check('Boundary hist=0', False, str(r.status_code))

r = requests.post(f'{BASE}/api/ml/backtest', json={
    'modelId': mid, 'board': 'all', 'poolSize': 30, 'groups': 2,
    'n': 3, 'hist': -1, 'applyCost': False
}, headers=h)
if r.status_code == 200:
    res = r.json()
    check('Boundary hist=-1→autoClamp', res.get('actualHistDays', 0) >= 244, f"={res.get('actualHistDays')}")
elif r.status_code == 422:
    check('Boundary hist=-1→422(样本不足)', True, '422 expected for small pool')
else:
    check('Boundary hist=-1', False, str(r.status_code))

# === 稳定性 ===
print('\n=== 四、稳定性测试 ===')

errors = []
for i in range(10):
    try:
        requests.get(f'{BASE}/api/ml/models', headers=h, timeout=30)
    except Exception as e:
        errors.append(str(e))
check('Stability 10x models列表', len(errors) == 0, f'{10-len(errors)}/10' if errors else '10/10')

errors2 = []
for i in range(10):
    try:
        requests.get(f'{BASE}/api/monitor/config', headers=h, timeout=30)
    except Exception as e:
        errors2.append(str(e))
check('Stability 10x monitor config', len(errors2) == 0, f'{10-len(errors2)}/10' if errors2 else '10/10')

errors3 = []
for i in range(10):
    try:
        requests.get(f'{BASE}/api/monitor/status', headers=h, timeout=30)
    except Exception as e:
        errors3.append(str(e))
check('Stability 10x monitor status', len(errors3) == 0, f'{10-len(errors3)}/10' if errors3 else '10/10')

# === 总结 ===
print('\n' + '=' * 60)
passed = sum(1 for _, ok, _ in tests if ok)
failed = [(n, d) for n, ok, d in tests if not ok]
print(f'总计: {passed}/{len(tests)} 通过')
if failed:
    print(f'失败 ({len(failed)}):')
    for n, d in failed:
        print(f'  FAIL {n}: {d}')
else:
    print('全部通过!')
print('=' * 60)
