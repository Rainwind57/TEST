# 简易量化研究平台

一个前后端分离的量化研究/交易模拟平台。后端基于 **FastAPI**，负责行情抓取、因子计算、选股、回测与模拟盘持久化；前端基于 **Vue 3 + Vite + Pinia + ECharts**，提供可视化界面。

## 功能模块

| 页面 | 路径 | 说明 |
|---|---|---|
| 行情 | `/quote` | 自选股实时行情、K 线展示 |
| 因子 | `/factor` | 单因子计算与查看 |
| 回归 | `/regression` | 因子与收益的回归分析 |
| 多因子回归 | `/factor-regression` | 多因子回归分析 |
| 选股 | `/select` | 按因子条件选股 |
| 分层回测 | `/backtest` | 因子分层回测 |
| 模拟盘 | `/portfolio` | 模拟交易、持仓、成交记录、净值曲线 |

## 项目结构

```
quant-platform/
├── backend/            # FastAPI 后端
│   ├── app/
│   │   ├── main.py         # 应用入口
│   │   ├── db.py           # SQLite 持久化（自选股/模拟盘）
│   │   ├── adapters.py     # 行情/K线数据源适配器
│   │   ├── factors.py      # 因子计算
│   │   └── routers/        # quote / factor / portfolio / selection 接口
│   ├── quant.db             # SQLite 数据文件（首次启动自动生成）
│   └── requirements.txt
└── frontend/           # Vue 3 前端
    ├── src/
    │   ├── views/           # 各功能页面
    │   ├── stores/          # Pinia 状态（自选股、模拟盘、提示）
    │   ├── api/client.js     # Axios 请求封装（默认请求 http://127.0.0.1:8899/api）
    │   └── router/           # 路由配置
    └── vite.config.js       # 开发端口 5899
```

## 环境要求

- Python 3.10+
- Node.js 18+

## 快速开始

### 1. 启动后端

```bash
cd quant-platform/backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8899
```

启动后访问 `http://127.0.0.1:8899/api/health` 应返回 `{"status": "ok"}`。首次启动会自动创建 `quant.db` 并初始化默认自选股与模拟盘（初始资金 100 万）。

### 2. 启动前端

```bash
cd quant-platform/frontend
npm install
npm run dev
```

启动后浏览器访问 `http://127.0.0.1:5899` 即可使用。

> 注意：前端固定请求后端 `http://127.0.0.1:8899/api`，请确保后端先启动并监听该端口，否则页面数据无法加载。

### 3. 生产构建（可选）

```bash
cd quant-platform/frontend
npm run build   # 产物输出到 dist/
npm run preview # 本地预览构建产物
```

## 常见问题

- **端口被占用**：后端端口 `8899`、前端端口 `5899` 均为固定端口，若被占用需修改 `frontend/vite.config.js`（前端端口）以及 `frontend/src/api/client.js` 的 `baseURL`（对应后端端口）后同步调整启动命令。
- **重置模拟盘**：模拟盘数据存储在 `backend/quant.db`，删除该文件后重启后端即可恢复到初始状态（初始资金 100 万，默认自选股）。
- **行情数据来源**：行情与 K 线数据通过服务端直连第三方接口（腾讯/东方财富等）获取，无需前端处理跨域问题；若接口失效需检查网络或更换数据源（见 `backend/app/adapters.py`）。
