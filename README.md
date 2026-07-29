# 简易量化研究平台

一个前后端分离的量化研究 / 交易模拟平台。后端基于 **FastAPI**，负责行情抓取、因子计算、选股、分层回测、机器学习、参数寻优、风险归因与模拟盘持久化；前端基于 **Vue 3 + Vite + Pinia + ECharts**，提供可视化界面。

## 功能模块

### 行情与因子
| 页面 | 路径 | 说明 |
|---|---|---|
| 行情 | `/quote` | 自选股实时行情、K 线展示 |
| 因子 | `/factor` | 单因子计算与查看 |
| 回归 | `/regression` | 因子与收益的回归分析 |
| 多因子回归 | `/factor-regression` | 多因子回归分析 |

### 选股与回测
| 页面 | 路径 | 说明 |
|---|---|---|
| 选股 | `/select` | 按因子条件选股，多因子加权打分 |
| 分层回测 | `/backtest` | 因子分层回测，含成本模型、基准对比、报告导出 |
| 分钟回测 | `/intraday` | 分钟级 K 线日内回测，止盈止损 |
| 模拟盘 | `/portfolio` | 模拟交易、持仓、成交记录、净值曲线 |

### 策略与研究闭环
| 页面 | 路径 | 说明 |
|---|---|---|
| 策略中心 | `/strategies` | 保存/重跑策略配置，自定义组合因子，回测存档 |
| 机器学习 | `/ml` | GBDT 因子收益预测，时序 Walk-Forward CV，特征重要性 |
| 参数寻优 | `/optimize` | Optuna 贝叶斯搜索回测参数，IS/OOS 分段验证 |
| 组合优化 | `/portfolio-opt` | 均值-方差 / 最大 Sharpe / 风险平价（cvxpy） |
| 风险归因 | `/risk` | Barra 风格风险分解，因子贡献 + 特质风险 |
| 盯盘调度 | `/monitor` | APScheduler 日频净值刷新 + 信号扫描 |
| 登录 | `/login` | 用户注册/登录，策略与存档按用户隔离 |

## 项目结构

```
quant-platform/
├── backend/                  # FastAPI 后端
│   ├── app/
│   │   ├── main.py             # 应用入口（路由注册 + 生命周期）
│   │   ├── db.py               # SQLite 持久化（自选股/模拟盘/策略/用户）
│   │   ├── adapters.py         # 行情/K线/资金流/北向/财务 数据源适配器
│   │   ├── factors.py          # 因子计算 + 中性化 + 绩效统计
│   │   ├── numpy_factors.py    # numpy 向量化因子（全序列版）
│   │   ├── ml.py                # 机器学习模块（GBDT + 时序 CV）
│   │   ├── optimize.py          # Optuna 参数寻优
│   │   ├── portfolio_opt.py    # 组合优化（cvxpy）
│   │   ├── risk.py              # Barra 风格风险归因
│   │   ├── intraday.py          # 分钟级回测
│   │   ├── scheduler.py         # APScheduler 盯盘调度
│   │   ├── auth.py             # JWT 鉴权
│   │   ├── jobs.py/jobs/ml/monitor/optimize/portfolio_opt/
│   │                           # data/risk/intraday/auth
│   ├── ml_models/              # 训练模型落盘目录（joblib）
│   ├── quant.db                # SQLite 数据文件（自动生成）
│   └── requirements.txt
└── frontend/                  # Vue 3 前端
    ├── src/
    │   ├── views/             # 各功能页面
    │   ├── stores/            # Pinia 状态（自选股/模拟盘/提示/鉴权）
    │   ├── api/client.js      # Axios 封装（默认 http://127.0.0.1:8899/api）
    │   └── router/            # 路由配置
    └── vite.config.js         # 开发端口 5899
```

## 环境要求

- Python 3.10+（开发使用 3.14）
- Node.js 18+

## 快速开始

### 1. 启动后端

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8899
```

> 若已将 Python `Scripts/` 目录加入 PATH，可直接使用 `uvicorn app.main:app ...`。
> `python -m uvicorn` 方式不依赖 PATH 配置，更稳妥。

启动后访问 `http://127.0.0.1:8899/api/health` 应返回 `{"status":"ok"}`。首次启动自动创建 `quant.db` 并初始化默认自选股与模拟盘（初始资金 100 万）。

### 2. 启动前端

```bash
cd frontend
npm install
npm run dev
```

浏览器访问 `http://127.0.0.1:5899`。

> 前端固定请求后端 `http://127.0.0.1:8899/api`，请确保后端先启动并监听该端口。

### 3. 生产构建（可选）

```bash
cd frontend
npm run build   # 产物输出到 dist/
npm run preview # 本地预览
```

## 数据源

行情与历史数据通过服务端直连第三方接口，无需前端处理跨域：

- **腾讯**：实时行情、日 K 线（前复权）、分钟 K 线
- **新浪**：全市场行情快照（分板块拉取）
- **东方财富**：资金流向、北向资金、财务指标

K 线采用三级缓存（内存 L1 → SQLite 磁盘 L2 → 网络），重复回测可减少 90%+ 网络请求。

## 核心能力

### 性能优化
- **向量化计算**：`numpy_factors.py` 用 numpy 重写 21 个因子，单因子计算提速 1-2 数量级
- **异步任务队列**：长回测改「提交 → job_id → 轮询」，避免前端 60s 超时
- **磁盘缓存**：K 线落盘 SQLite，仅增量补抓新交易日

### 防过拟合机制
- **Walk-Forward CV**：ML 与参数寻优均用前半样本调参、后半样本验证
- **Purged 分割**：训练/测试间留 gap 防信息泄漏
- **IS/OOS 报告**：寻优结果同时回报两段指标

### 风险与归因
- **因子中性化**：对风格暴露（市值/行业 OLS 取残差，消除风格偏差
- **Barra 归因**：组合收益分解为风格因子贡献 + 特质残差，风险分解为因子风险 + 特质风险

## 多用户

轻量 SaaS：用户表 + JWT 鉴权，策略/自定义因子/回测存档按 `user_id` 隔离。行情与模拟盘保持单机共享。匿名访问兼容（user_id=0）。

## 常见问题

- **`uvicorn` 命令未找到**：用 `python -m uvicorn ...` 启动，或将 Python `Scripts/` 目录加入 PATH。
- **端口被占用**：后端 `8899`、前端 `5899` 为固定端口，需同步修改 `frontend/vite.config.js` 与 `frontend/src/api/client.js` 的 `baseURL`。
- **重置数据**：删除 `backend/quant.db` 后重启后端，恢复初始状态。模型落盘在 `backend/ml_models/`，可手动清理。
- **数据源失效**：检查网络或更换数据源（见 `backend/app/adapters.py`）。腾讯接口偶发限流，重试即可。
- **依赖安装慢**：可配置国内镜像 `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`。

## 技术栈

**后端**：FastAPI · SQLite · httpx · numpy · pandas · scikit-learn · lightgbm · optuna · cvxpy · apscheduler · joblib

**前端**：Vue 3 · Vite · Pinia · Vue Router · ECharts · Axios
