# TEST-main (1) 新版量化平台 · 优化空间审计报告

> 审计对象：`C:\Users\asus\Downloads\TEST-main (1)\TEST-main`（FastAPI 后端 27 模块 + Vue3 前端 15 页面）
> 审计时间：2026-07-30 ｜ 方式：前后端全量源码审查（双审计线并行），所有问题定位到 文件:行号
> 背景：本版是在上一轮审计（31 条 bug）基础上迭代的版本。本报告先验证用户实测的三大痛点根因，再列出新发现的问题，最后给出修复路线。

---

## 〇、总体结论（先看这里）

| 维度 | 结论 |
|---|---|
| 上版 P0「模型训练后无人消费」 | **部分修复**：ML 页新增了专用模型回测/打分（`joblib.load` + 特征同构 + 预处理参数随 bundle 保存，做得对）——但**只修了一半** |
| 用户痛点① 回测无法导入模型 | **属实**。主回测入口（分层回测）仍然只认技术因子，`modelId` 无门可入 → 形成"两套互不相通的回测体系" |
| 用户痛点② 因子数不足 | **属实**。技术因子 22 个 + 快照因子 14 个，无一致预期/资金流趋势/质量类因子；且 ML 训练默认只喂 22 个技术因子 |
| 用户痛点③ 网络连接不好 | **属实且有明确技术根因**：行情列表单一依赖新浪无降级、大部分外部请求无重试、**空结果被缓存 300 秒**、主回测前端 60s 超时误杀长任务 |
| 上版数值类 bug（年化×n、Sharpe×√n、oosSharpe恒0、ICIR口径、协方差用错） | **均已修复** ✅（本次逐条复核确认） |
| 新发现问题 | 共 **24 条**（P0×4 / P1×9 / P2×11），详见下文 |

---

## 一、痛点① 根因：回测无法导入模型

### 1.1 模型侧其实已经打通（值得肯定）

- `ml.py:281-286`：模型落盘 bundle 含 `model + feature_names + preprocess`（缩尾分位数、列均值/标准差），训练与推理特征**同名同序同标准化**，上版"特征错位"隐患已消除。
- `ml.py:417-425` `_load_model` 有 FileNotFoundError 兜底；`routers/ml.py:96-109`（`/api/ml/backtest`）与 `routers/jobs.py:101-110`（`ml-backtest` job）能正确把 `modelId` 透传给 `ml.backtest_model`。

**即：模型"能"回测——但只能在 ML 页那条隐蔽的专用通道里。**

### 1.2 断点一（后端 P0）：主回测入口不接受模型

- `routers/selection.py:222-236`：主回测请求体 `BacktestBody` **没有 `modelId` 字段**。
- `routers/selection.py:240`：
  ```python
  if body.factor not in FACTORS:
      raise HTTPException(400, "分层回测目前仅支持技术类(量价)因子")
  ```
  强制只接受内置技术因子。即使前端硬塞 modelId 也会被忽略。

### 1.3 断点二（前端 P0）：主回测页没有模型入口

- `BacktestView.vue:28,46,171-172`：表单只有 `factorKey`，因子下拉只拉技术因子；`L51-57` 请求体无 `modelId`；`L59-66` 只调 `/select/backtest`。**主回测页 100% 无法选模型。**
- `MLView.vue:100-113`：模型回测按钮存在，但结果只展示在 ML 页本地卡片（L233-243），**无导出、无"去主回测页"、从未写入共享 store**。

### 1.4 断点三（P1）：寻优也吃不到模型

- `optimize.py:33-55`：`_objective` 写死 `sel.run_backtest(sel.BacktestBody(**cfg))`，参数寻优只能优化单因子策略，**模型策略无法寻优**。

### 1.5 断点四（P1）：网络失败伪装成"无法回测"

- `ml.py:509-520`：单股 K 线拉取失败静默返回 `[]`，有效股票 `< groups×3` 时抛"有效股票样本不足"。**网络一差，用户看到的报错是"样本不足"，体感就是"模型回测不了"**——痛点①和痛点③在这里叠加。

### 1.6 附带口径问题（P2）

- 训练截面从 `i=60` 起（`ml.py:67`），回测从 `t=25` 起（`ml.py:565`）：长窗口因子（momentum120 等）在回测早期被整股丢弃，早期与后期有效特征集不一致。建议回测起点统一为 `max(60, 最大因子窗口)`。

---

## 二、痛点② 根因：因子数不足

### 2.1 现状盘点

| 类别 | 数量 | 明细 |
|---|---|---|
| 技术量价因子（`factors.py:367-389`） | 22 | momentum5/10/20/60/120、ma_dev/ma5_dev/ma60_dev、volatility/volatility60、rsi/rsi6、macd、kdj_k、wr14、cci14、boll_pct、amplitude20、volume_ratio5、obv_trend、high_low_pos、dist_52w_high |
| 快照因子（`factors.py:394-409`） | 14 | pct_chg、turnover、amount、pe、pb、ep、bp、mkt_cap、circ_mkt_cap、roe、net_margin、revenue_yoy、profit_yoy、main_net_pct |

### 2.2 缺口与"白拉的数据"

1. **已拉到却没接线的财务字段（P1，改造成本最低）**：`adapters.py:441-466` `fetch_finance_summary` 已经拉取 `grossMargin / debtRatio / eps / bps`，但**未注册进 SNAPSHOT_FACTORS**——数据白白拉了。
2. **ML 吃不到大部分快照因子（P1）**：训练默认 `use_snapshot=False`；即便打开，也只追加 pe/pb/turnover 三个（`ml.py:47,80`），ep/bp/mkt_cap/roe 等进不了模型，且这三个是**静态快照回填历史截面，含前视偏差**。
3. **完全缺失的类别（P2）**：
   - 质量类：ROA、现金流质量/应计、股息率
   - 资金流：只有 `main_net_pct` 单点，无连续资金流趋势；已有 `fetch_north_flow_trend` 却无北向因子
   - 一致预期：分析师评级/盈利预测修正——数据源层面就没有，需新增接口

---

## 三、痛点③ 根因：网络连接不好

按影响从大到小：

1. **行情列表单一依赖新浪、无降级（P0）**：`adapters.py:243-277` `fetch_market_list` 硬编码新浪 `Market_Center`，无腾讯/东财 fallback。新浪限流 → 股票池为空 → 训练/回测全线报"样本不足"。**这是"网络不好"的第一根因。**
2. **空结果被缓存 300 秒（P0）**：`adapters.py:315-318` 限流返回的空 K 线也写入缓存，此后 5 分钟内所有调用都拿到空数据，故障被放大 300 秒。应跳过空结果写缓存。
3. **重试覆盖不全（P1）**：只有 `_http_get`（`adapters.py:25-37`，2 次重试+退避）用于腾讯行情/K线；新浪行情、东财资金流/财务全部裸 `client.get`，10s 超时直接抛。
4. **前端 60s 超时误杀长回测（P0，前端）**：`api/client.js:8` 默认 timeout 60s，`BacktestView` 直接用默认值跑同步回测；而 MLView 已用 5 分钟 `longTask`。**长回测在弱网/大股池下会被前端超时掐断，体感即"网络连接不好"。**
5. **CORS 硬编码（P1）**：`main.py:14` 只允许 `localhost:5899`，换 IP/端口访问直接被浏览器拦，体感也是"连不上"。
6. **轮询无终止（P1，前端）**：`MLView.vue:39-51` `pollJob` 是 `while(true)`，无 `onUnmounted` 停止、无 AbortController，离开页面后仍在后台空转。
7. **定时刷新无在途锁（P2，前端）**：QuoteView/PortfolioView 每 6s `setInterval` 拉行情，弱网下请求堆积重叠。

---

## 四、其他新发现问题清单

### 后端

| # | 级别 | 位置 | 问题 |
|---|---|---|---|
| B1 | **P0** | `auth.py:23` | JWT 默认弱密钥：未设 `QUANT_JWT_SECRET` 时用固定字符串，token 可伪造 |
| B2 | P1 | `ml.py:204-205,276` | `model.fit` 在 async 函数内同步执行，阻塞事件循环，训练期间全站请求卡死；应 `run_in_executor`（optimize 模块已有正确写法可参考） |
| B3 | P1 | `routers/auth.py:46-48` | 除 reset/me 外所有接口不强制登录，模拟盘下单接口完全开放 |
| B4 | P1 | `ml.py:60`、`selection.py:100,264` 等 | 多处 `except Exception: continue` 吞异常，把网络失败伪装成"样本不足"，无法排障 |
| B5 | P2 | `ml.py:598` | 持仓期内停牌未处理：跨停牌用缺口价算收益，收益失真 |
| B6 | P2 | `jobs.py:10-11` | job 表在进程内存中，重启即丢、多 worker 不共享 |
| B7 | P2 | `adapters.py` 解析层 | 缺失字段默认填 0.0，与真实 0 混淆，建议用 None |

### 前端

| # | 级别 | 位置 | 问题 |
|---|---|---|---|
| F1 | P1 | `stores/research.js:13-14` | `setCurrentModel` / `setCurrentSelectResult` 全仓无调用点——跨页共享桥只建了一半（寻优→回填那半是通的） |
| F2 | P1 | `FactorView` / `RegressionView` / `FactorRegressionView` | 算完即死路：无任何"下一步/去回测"入口（无 router.push） |
| F3 | P1 | `SelectView.vue` | 选股结果只能加自选/买入模拟盘，**无"用该股池回测"** |
| F4 | P2 | `PortfolioOptView.vue:46-58` | 优化权重可一键下单（这点新版打通了✅），但只 toast 不跳转模拟盘页 |
| F5 | P2 | `BacktestView.vue:53-56` | 表单清空后 `Number('')=0` 发出脏请求，无校验拦截 |
| F6 | P2 | `MLView` btResult | ML 回测结果无导出能力（主回测页有 HTML/Excel 导出✅） |
| F7 | P2 | `api/client.js:20-36` | 401 整页跳转、错误 toast 1.8s 自动消失且无重试按钮/详情 |
| F8 | P2 | `Sidebar.vue:58-68` | 移动端固定 200px 侧栏遮挡正文，无汉堡菜单；宽表无横向滚动 |
| F9 | P2 | 全站 | 无加载骨架屏，仅按钮文案变化 |

### 已复核确认修复（上版遗留 → 新版 ✅）

年化口径 `ppy=252/n`（`selection.py:381`）、日频 Sharpe/OOS Sharpe（`ml.py:371-393`）、ICIR `√ppy`（`factors.py:589-597`）、purged walk-forward（`ml.py:153-172`）、多空成本 `2×cost`（`selection.py:352`）、Barra 协方差改用因子收益时序（`risk.py:87-98`）、涨跌停约束（`selection.py:324-328`）、SQLite WAL+BEGIN IMMEDIATE 防双花（`db.py:16-18`、`routers/portfolio.py:94`）、幸存者偏差警告（`selection.py:432`）、密码 PBKDF2 十万次+参数化 SQL。

---

## 五、修复路线图（按投入产出排序）

### 第一阶段：打通模型主链路（对应痛点①，约 1~2 天）
1. `routers/selection.py` `BacktestBody` 增加可选 `modelId`；`run_backtest` 加分支：有 modelId 时复用 `ml.backtest_model`（或统一两条回测路径为一个实现）。
2. `BacktestView.vue` 增加"策略来源"切换（技术因子 / ML 模型下拉，拉 `/ml/models`），请求体带 `modelId`，并改走 jobs 异步 + 轮询（顺带解决 60s 超时误杀）。
3. `MLView` 训练完成后写 `research.setCurrentModel(m)` 并提供"去主回测"按钮 → BacktestView `onMounted` 消费。
4. `optimize.py` 支持 modelId 透传，让模型策略也能寻优。

### 第二阶段：网络健壮性（对应痛点③，约 1~2 天）
5. `fetch_market_list` 增加腾讯/东财 fallback；所有外部 GET 统一走带重试/退避的 `_http_get`。
6. 空 K 线/空列表**禁止写缓存**；缓存命中空值时强制重拉。
7. CORS 白名单环境变量化；`pollJob` 加 `onUnmounted` 终止；行情轮询加在途锁。

### 第三阶段：因子扩容（对应痛点②，约 2~3 天）
8. 把已拉取的 `grossMargin/debtRatio/eps/bps` 注册进 SNAPSHOT_FACTORS（零新增接口成本）。
9. ML 训练支持全部快照因子并明示前视风险（或改造成 PIT 对齐）。
10. 新增北向资金趋势因子（`fetch_north_flow_trend` 已有）；评估接入一致预期数据源。

### 第四阶段：工程与体验（持续）
11. `QUANT_JWT_SECRET` 强制环境变量 + 写操作全局鉴权。
12. ML 训练移入 `run_in_executor`；job 表落 SQLite。
13. 回测起点与训练窗口对齐；持仓期停牌处理。
14. 死路页面补"下一步"按钮；错误提示加重试；骨架屏；移动端侧栏。

---

## 附：本次与上一版审计的关系

- 上版报告《TEST-main程序Bug清单与竞品对比分析.md》给出 31 条问题与竞品对比（聚宽/BigQuant/Qlib/米筐/QMT 十维度），本版已修其中**全部数值正确性类**与部分工程类问题，方向正确。
- 剩余最大差距仍是**闭环**：Qlib 用一份 workflow 配置贯通"训练→预测→组合→回测"，BigQuant 用画布连线让死路在 UI 层不可能出现。本版把模型回测做成了 ML 页的"独立景点"，而竞品是把模型作为主回测的"一等公民策略源"——第一阶段的 4 条改动（后端约 80 行 + 前端一个下拉框）即可对齐这个设计。
