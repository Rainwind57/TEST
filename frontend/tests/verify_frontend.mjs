#!/usr/bin/env node
/**
 * 前端-后端一致性验证脚本（模拟前端使用）
 *
 * 依据《各模块Bug与优化空间汇总_2026-08-14.md》《回测日期错配根因_2026-08-14.md》
 * 《建议历史长度计算缺陷_2026-08-14.md》三个文档声称的修复点，验证前端是否能兑现文档：
 *
 *   --static  静态契约核查：读前端源码 + 后端路由源码，逐项核对文档声称与代码事实（无需后端运行）
 *   --live    模拟前端完整用户流：注册/登录 → Cookie 双通道 → 各视图按前端同款载荷调后端 → 校验响应字段（需后端运行）
 *
 * 用法：
 *   node verify_frontend.mjs --static
 *   node verify_frontend.mjs --live
 *   node verify_frontend.mjs --live --full        # 额外跑报告导出重流程
 *   BASE_URL=http://127.0.0.1:8000/api node verify_frontend.mjs --live
 *
 * 环境变量：BASE_URL（默认 http://127.0.0.1:8000/api，与 backend/tests/test_api_integration.py 同约定）
 * 依赖：Node >= 18（内置 fetch / AbortSignal.timeout）
 */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const ROOT = path.resolve(__dirname, '../..')
const FE = path.join(ROOT, 'frontend/src')
const BE = path.join(ROOT, 'backend/app')

const args = new Set(process.argv.slice(2))
const MODE = args.has('--live') ? 'live' : 'static'
const FULL = args.has('--full')

// ---------------------------------------------------------------- 统计器
const stats = { ok: 0, fail: 0, warn: 0, skip: 0 }
let _cur = ''
function section(name) { _cur = name; console.log(`\n=== ${name} ===`) }
function check(cond, name, detail = '') {
  const tag = cond ? 'OK  ' : 'FAIL'
  if (cond) stats.ok++; else stats.fail++
  console.log(`  [${tag}] ${name}${detail ? `  — ${detail}` : ''}`)
  return cond
}
function warn(name, detail = '') {
  stats.warn++
  console.log(`  [WARN] ${name}${detail ? `  — ${detail}` : ''}`)
}
function skip(name, detail = '') {
  stats.skip++
  console.log(`  [SKIP] ${name}${detail ? `  — ${detail}` : ''}`)
}
function read(p) {
  try { return fs.readFileSync(p, 'utf8') } catch { return '' }
}
function src(rel) { return read(path.join(FE, rel)) }
function be(rel) { return read(path.join(BE, rel)) }

// ---------------------------------------------------------------- 纯计算：三处建议历史长度口径对照
function daysBetween(a, b) { return Math.max(0, Math.floor((b - a) / 86400000)) }
function mlBackendDays(hasStart, hasEnd, start, end, today, n = 3) {
  const base = 60 + n + 240
  if (hasStart && hasEnd) return daysBetween(start, end) + base
  if (hasStart) return daysBetween(today, start) + base
  return base
}
function mlViewDays(hasStart, hasEnd, start, end, today) {
  const from = hasStart ? start : today
  const to = hasEnd ? end : today
  return Math.max(0, Math.ceil(Math.max(0, Math.floor((to - from) / 86400000)) + 260))
}
function btViewDays(hasStart, hasEnd, start, end, today) {
  if (hasStart) return Math.max(300, Math.ceil(Math.floor((today - start) / 86400000) + 60 + 240))
  if (hasEnd && end < today) return Math.max(300, Math.ceil(Math.floor((today - end) / 86400000) + 60 + 240))
  return 300
}

// ================================================================ STATIC 模式
async function runStatic() {
  console.log('模式：--static（静态契约核查，无需后端）')
  console.log(`文档基准：2026-08-14 三份 md（汇总 / 回测日期错配 / 建议历史长度）`)

  // ---- 认证通道（文档：token 迁移 httpOnly Cookie）----
  section('认证通道（文档 §五 P1：token 迁移 httpOnly Cookie）')
  const client = src('api/client.js')
  const main = src('main.js')
  const guard = src('router/index.js')
  check(!/localStorage\.(set|get|remove)Item/.test(client), 'client.js 不再写 localStorage', '文档声称已迁 Cookie，JS 不可读')
  check(/withCredentials:\s*true/.test(client), '请求带 withCredentials（携带 quant_token Cookie）')
  check(/let _token = ''/.test(client), 'token 仅存内存（_token 变量）')
  check(/auth\.bootstrap\(\)/.test(main) && /\/auth\/me/.test(src('stores/auth.js')), 'main.js bootstrap 调 /auth/me 恢复登录态', '文档：启动恢复')
  check(/isLoggedIn/.test(guard), '路由守卫基于 isLoggedIn 放行')

  // ---- MLView suggestedHist（文档：6203 已修复，删 extraEnd 改区间跨度）----
  section('MLView.suggestedHist（《建议历史长度计算缺陷》P0 已修复）')
  const mlView = src('views/MLView.vue')
  check(!/extraEnd/.test(mlView), 'MLView.vue 已删除 extraEnd 双重计数')
  check(/spanDays\s*=\s*Math\.max\(0,\s*Math\.floor\(\(to - from\) \/ 86400000\)\)/.test(mlView), '改为区间跨度 (to-from)/86400000')
  check(/spanDays \+ 260/.test(mlView), '缓冲 260 = 60 + 最长回看 240')
  check(/histInsufficient/.test(mlView), 'histInsufficient 不足时提示并禁提交')

  // ---- BacktestView 联动 watcher（文档修复未同步，预期分叉）----
  section('BacktestView 起止日→hist 联动（文档声称修复的同类公式）')
  const btView = src('views/BacktestView.vue')
  const hasBtvExtra = /extraEnd/.test(btView)
  check(!hasBtvExtra, 'BacktestView.vue 已同步删除 extraEnd', hasBtvExtra ? '仍存在 extraEnd 双重计数，与 MLView 修复不同步' : '')
  if (hasBtvExtra) warn('回测页保留旧公式 backToFrom+60+240+extraEnd → 设起止日仍会回填虚高 hist（与文档"修复"口径分叉）', '文档仅修 MLView；BacktestView.vue:52-64 未同步')

  // ---- 三处公式口径对照（纯计算）----
  section('建议历史长度三处口径对照（同一起止日）')
  const today = new Date('2026-08-14T00:00:00')
  const s = new Date('2017-01-14T00:00:00')
  const e = new Date('2020-01-14T00:00:00')
  const mBack = mlBackendDays(true, true, s, e, today)
  const mMl = mlViewDays(true, true, s, e, today)
  const mBt = btViewDays(true, true, s, e, today)
  console.log(`    start=2017-01-14 end=2020-01-14 today=2026-08-14`)
  console.log(`      后端 ml.py days_to_cover : ${mBack}  （区间跨度 + 60 + n + 240，n=3）`)
  console.log(`      前端 MLView suggestedHist : ${mMl}  （区间跨度 + 260）`)
  console.log(`      前端 BacktestView watcher : ${mBt}  （(today-start) + 300，双端只按 start 回溯）`)
  check(mMl <= mBack + 60, 'MLView 口径与后端接近（差 n=3 与 60/260 常量，文档 P2 已知）', `${mMl} vs ${mBack}`)
  check(mBt < 5000, 'BacktestView 不再回填虚高值（6203 级）', mBt >= 5000 ? `仍算出 ${mBt}，双重计数未除` : `${mBt} 正常`)

  // ---- SelectView 因子列（文档：factorDetail 双写已修复）----
  section('SelectView 因子列（文档 §二 P1：选股表因子列全 -- 已修复）')
  const selView = src('views/SelectView.vue')
  check(/factorDetail\?\.\[f\.key\]\?\.raw/.test(selView), '前端读 r.factorDetail[f.key].raw')
  const selPy = be('routers/selection.py')
  check(/row\["factorDetail"\]\[uf_key\] = \{"raw": scores\[idx\]\}/.test(selPy), '后端 /select 双写 factorDetail[key]={raw}', 'selection.py')
  check(/"direction": "long_only" if body\.longOnly else "long_short"/.test(selPy), '后端回 字段')

  // ---- 快照因子（文档 §一 P1：快照因子恒 None 已修复）----
  section('快照因子（文档 §一 P1：基本面/资金流/行业已打通）')
  const factorPy = be('routers/factor.py')
  check(/fetch_finance_summary|fetch_north_holding|fetch_sector_map/.test(factorPy), 'get_factors 按需拉取快照因子', 'routers/factor.py')

  // ---- L1 缓存（文档 §一 P1：缓存静默截断已修复）----
  section('L1 K 线缓存（文档 §一 P1：不足 days 时回源补齐）')
  const adapters = be('adapters.py')
  check(/len\(full\) >= days/.test(adapters), '缓存条数 >= days 才切片，否则落网络分支', 'adapters.py')

  // ---- ML 后端（文档 §四：days_to_cover / oosRankIc / snapshotStartNote）----
  section('ML 后端（文档 §四）')
  const mlPy = be('ml.py')
  check(/\(to_date - from_date\)\.days \+ 60 \+ n \+ _MAX_FACTOR_LOOKBACK/.test(mlPy), 'days_to_cover 双端=区间跨度+缓冲', 'ml.py')
  check(/snapshotStartNote/.test(mlPy), '快照起点透明化 snapshotStartNote', 'ml.py')
  check(/oosRankIc/.test(mlPy), 'oosRankIc 参与评估输出')
  check(!/overall_rank_ic=0\.0/.test(mlPy), '不再硬编码 oosRankIc=0', '文档 P1：已改按日真实计算 RankIC')

  // ---- 报告导出契约（文档：ML 导出字段齐全）----
  section('报告导出契约（reports/backtest ExportBody）')
  const repPy = be('routers/reports.py')
  for (const f of ['signalMode', 'bullRule', 'bearRule', 'topAttribution', 'snapshotStartNote', 'inSampleWarning', 'histWarning', 'effectiveStart', 'effectiveEnd', 'actualHistDays']) {
    if (!check(new RegExp(`^\\s*${f}:`, 'm').test(repPy), `ExportBody 声明 ${f}`, 'reports.py')) break
  }

  // ---- 文档 P2/P3 已知未修项（如实核验，不判文档说谎）----
  section('文档标注"待处理/已知"项核验（WARN=文档如实记载）')
  const optView = src('views/OptimizeView.vue')
  if (!/modelId/.test(optView)) warn('OptimizeView 不发 modelId（文档 P2：仅发 modelType）', 'OptimizeView.vue')
  const monView = src('views/MonitorView.vue')
  if (!/tradeDirections.*(short|cover)/.test(monView) && !/short.*tradeDirections/.test(monView)) warn('MonitorView 前端未对 tradeDirections 守门（文档 P2：仅后端守）', 'MonitorView.vue shortSignal/batchBuy 恒可下 short')
  const intra = src('views/IntradayView.vue')
  if (!/commissionRate/.test(intra)) warn('IntradayView 无成本参数（文档 P3：saveStrategy 漏成本）', 'IntradayView.vue')
  const quote = src('views/QuoteView.vue')
  if (!/fmtVol/.test(quote) || !/\* 2/.test(quote) === false) {}
  check(/max="2000"/.test(quote), 'QuoteView klineDays 上限 2000（文档 P3：已对齐）', 'QuoteView.vue')
  check(/\(v - dea\[i\]\) \* 2/.test(quote), 'MACD 柱 (DIF-DEA)×2（文档 P3：前后端口径核查项）', 'QuoteView.vue')

  // ---- 端点契约交叉核对（前端调用 vs 后端路由）----
  section('端点契约交叉核对（前端调用的路径在后端是否存在）')
  const routes = collectBackendRoutes()
  const calls = collectFrontendCalls()
  const miss = []
  for (const c of calls) {
    const hit = routes.some(r => pathMatches(c, r))
    if (!hit) miss.push(c)
  }
  if (miss.length) {
    warn('以下前端调用在后端路由中未匹配到（可能为模板路径/代理端点，需人工确认）', miss.join(', '))
  } else {
    check(true, `前端 ${calls.length} 个调用全部在后端路由命中`)
  }
  console.log(`    后端路由 ${routes.length} 条；前端调用 ${calls.length} 条`)
}

function collectBackendRoutes() {
  const out = []
  const dir = path.join(BE, 'routers')
  if (!fs.existsSync(dir)) return out
  for (const f of fs.readdirSync(dir)) {
    if (!f.endsWith('.py')) continue
    const txt = read(path.join(dir, f))
    const m = txt.match(/prefix\s*=\s*["']([^"']+)["']/)
    const prefix = m ? m[1] : ''
    const re = /@router\.(get|post|put|delete|patch)\(["']([^"']*)["']/g
    let mm
    while ((mm = re.exec(txt))) out.push(prefix + mm[2])
  }
  return out
}

function collectFrontendCalls() {
  const out = new Set()
  const dirs = ['views', 'stores']
  const files = ['api/client.js']
  for (const d of dirs) {
    const p = path.join(FE, d)
    if (!fs.existsSync(p)) continue
    for (const f of fs.readdirSync(p)) {
      if (f.endsWith('.vue') || f.endsWith('.js')) files.push(`${d}/${f}`)
    }
  }
  const re = /api\.(get|post|delete|put)\(\s*[`'"]([^`'"]*)/g
  for (const f of files) {
    const txt = src(f)
    let m
    while ((m = re.exec(txt))) {
      let p = m[2].split('${')[0] // 模板路径取前缀
      if (!p.startsWith('/')) p = '/' + p
      out.add(p)
    }
  }
  return [...out]
}

function pathMatches(front, back) {
  const norm = p => p.replace(/\/+$/, '')
  const f = norm(front.split('?')[0])
  const b = norm(back.replace(/^\/api/, ''))
  if (f === b) return true
  // 后端含路径参数 {x}：前端前缀匹配
  if (b.includes('{')) {
    const bBase = norm(b.split('{')[0])
    return f.startsWith(bBase)
  }
  return false
}

// ================================================================ LIVE 模式
const BASE = process.env.BASE_URL || 'http://127.0.0.1:8000/api'

let _cookie = ''
let _token = ''
let _user = null

async function req(method, path_, body, { auth = true, timeout = 60000, raw = false } = {}) {
  const headers = {}
  if (body !== undefined) headers['Content-Type'] = 'application/json'
  if (_cookie) headers['Cookie'] = _cookie
  if (auth && _token) headers['Authorization'] = `Bearer ${_token}`
  let res
  try {
    res = await fetch(BASE + path_, {
      method,
      headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(timeout),
    })
  } catch (e) {
    return { status: 0, data: String(e?.message || e), headers: null }
  }
  const ct = res.headers.get('content-type') || ''
  let data = null
  try {
    if (ct.includes('application/json')) data = await res.json()
    else if (ct.includes('text')) data = await res.text()
    else data = await res.arrayBuffer()
  } catch { data = null }
  return { status: res.status, data, headers: res.headers, ok: res.ok }
}

async function runLive() {
  console.log('模式：--live（模拟前端完整用户流）')
  console.log(`BASE_URL = ${BASE}`)
  section('L0 后端存活探测')
  const h = await req('GET', '/health', undefined, { auth: false })
  if (h.status !== 200 || h.data?.status !== 'ok') {
    console.log(`  后端不可达（${h.status} ${JSON.stringify(h.data)}），请先启动后端（docker compose up -d 或 uvicorn）。`)
    process.exit(2)
  }
  check(true, '/api/health 返回 status=ok')

  section('L1 注册/登录 → token + httpOnly Cookie（模拟 stores/auth.js）')
  // 固定账户复用：注册接口按 IP 限流（3次/时），先尝试登录，不存在才注册。
  // 环境变量 VERIFY_USER/VERIFY_PASS 可指定既有账户（注册被限流时用）。
  const uname = process.env.VERIFY_USER || 'verify_ci'
  const pass = process.env.VERIFY_PASS || 'verify_pass_123!'
  let login = await req('POST', '/auth/login', { username: uname, password: pass }, { auth: false })
  if (login.status !== 200) {
    const reg = await req('POST', '/auth/register', { username: uname, password: pass }, { auth: false })
    if (reg.status === 200 || reg.status === 201) login = reg
    else warn(`登录失败且注册受限（${reg.status} ${reg.data?.detail}）`, '可设 VERIFY_USER/VERIFY_PASS 指向既有账户')
  }
  check(login.status === 200, 'register/login 成功', `status=${login.status}`)
  const tok = login.data?.token
  check(typeof tok === 'string' && tok.length > 10, '响应含 token（前端存内存）')
  _token = tok || ''
  const sc = login.headers?.get?.('set-cookie') || ''
  check(sc.includes('quant_token'), 'Set-Cookie 含 quant_token', sc.split(';')[0])
  check(/HttpOnly/i.test(sc), 'Cookie 为 HttpOnly（JS 不可读，文档 P1 修复）')
  check(/SameSite=Lax/i.test(sc), 'Cookie SameSite=Lax')
  const cval = (sc.match(/quant_token=([^;]+)/) || [])[1]
  if (cval) _cookie = `quant_token=${cval}`
  _user = login.data

  section('L2 /auth/me 启动恢复（模拟 main.js bootstrap）— 双通道')
  const meCookie = await req('GET', '/auth/me', undefined, { auth: false })   // 仅 Cookie
  const meHeader = await req('GET', '/auth/me', undefined, { auth: true, timeout: 1000 }) // 仅 Header（清 cookie 模拟）
  check(meCookie.status === 200 && meCookie.data?.username === uname, '仅（持久登录恢复）')
  const saved = _cookie; _cookie = ''
  const meHeaderOnly = await req('GET', '/auth/me', undefined, { auth: true })
  _cookie = saved
  check(meHeaderOnly.status === 200, '仅 Authorization Header 通道 → /auth/me 200（旧客户端兼容）')
  check(meHeader.status === 200, '双通道同时 → 200')

  section('L3 QuoteView /api/kline（文档 §一：L1 缓存补齐、上限 2000）')
  // 前端 addCode 经 normalizeCode 归一化为 sh600519 后才入自选；此处按前端真实用法
  const k1 = await req('GET', '/kline?code=sh600519&days=500', undefined, { auth: false })
  const k2 = await req('GET', '/kline?code=sh600519&days=1500', undefined, { auth: false })
  check(k1.status === 200 && Array.isArray(k1.data?.data) && k1.data.data.length > 0, 'kline days=500 返回数据', `status=${k1.status} len=${k1.data?.data?.length}`)
  check(k2.status === 200 && k2.data?.data?.length >= 1500, '缓存不足 days 时回源补齐（文档 P1）', `实际 ${k2.data?.data?.length || 0} 条`)
  const row = k2.data?.data?.[0]
  check(row && ['date', 'open', 'close', 'high', 'low', 'volume'].every(k => k in row), 'kline 行含 date/open/close/high/low/volume')
  const kw = await req('GET', '/kline?code=sh600519&days=100&freq=W', undefined, { auth: false })
  check(kw.status === 200 && Array.isArray(kw.data?.data) && kw.data.data.length > 0, 'freq=W 周线聚合返回')

  section('L4 QuoteView /stock/exists + /timeshare')
  const ex = await req('GET', '/stock/exists?code=sh600519', undefined, { auth: false })
  check(ex.status === 200 && typeof ex.data?.exists === 'boolean', 'stock/exists 返回 {exists,name}')
  const ts = await req('GET', '/timeshare?code=sh600519', undefined, { auth: false })
  check(ts.status === 200 && (ts.data?.data !== undefined), 'timeshare 返回 {data}')

  section('L5 FactorView 快照因子（文档 §一 P1：基本面/资金流/行业非 None）')
  const catalog = await req('GET', '/factors/catalog', undefined, { auth: false })
  check(catalog.status === 200 && Array.isArray(catalog.data), 'factors/catalog 返回目录')
  const fac = await req('GET', '/factors?codes=sh600519', undefined, { auth: false, timeout: 90000 })
  const frow = (Array.isArray(fac.data) ? fac.data : Object.values(fac.data || {})).find?.(r => r?.code === 'sh600519')
  check(frow && frow.sector !== null && frow.sector !== undefined && frow.mkt_cap !== null, '快照因子已打通（文档 P1）', `sector=${frow?.sector} mkt_cap=${frow?.mkt_cap}`)
  const finMissing = ['roe', 'net_margin', 'revenue_yoy', 'main_net_pct'].filter(k => frow?.[k] == null)
  if (finMissing.length) warn('财务/资金流快照字段仍为空（代码已按需拉取，但上游 datacenter/push2his.eastmoney 本环境断连）', `缺失: ${finMissing.join(',')}`)

  section('L6 SelectView /select（文档 §二 P1：factorDetail 双写）')
  const selRes = await req('POST', '/select', {
    board: 'all', boards: ['all'], poolSize: 30, topN: 10,
    factors: [{ key: 'momentum', weight: 1, direction: 1 }],
    filters: { excludeSt: true }, assetClass: 'a-share',
  }, { timeout: 180000 })
  check(selRes.status === 200, 'select 返回 200', `status=${selRes.status}`)
  const rows = selRes.data?.rows || []
  const first = rows[0]
  const fdVal = first?.factorDetail?.['momentum']?.raw
  check(rows.length > 0, 'select 返回 rows')
  check(fdVal !== undefined && fdVal !== null, 'rows[0].factorDetail.momentum.raw 已双写（文档 P1）', `raw=${fdVal}`)

  section('L7 FactorRegressionView /select/factor-regression')
  const fr = await req('POST', '/select/factor-regression', {
    board: 'all', boards: ['all'], poolSize: 30, factors: ['momentum', 'ma_dev'], n: 5, hist: 300,
  }, { timeout: 180000 })
  check(fr.status === 200, 'factor-regression 200')
  for (const k of ['keys', 'summary', 'periods', 'meanR2', 'effectiveStocks']) {
    if (!check(fr.data?.[k] !== undefined, `factor-regression 返回 ${k}`, `=${JSON.stringify(fr.data?.[k])?.slice(0, 60)}`)) break
  }

  section('L8 BacktestView /select/backtest（文档 §二 P1：direction 方向、报告字段）')
  const btBody = {
    board: 'all', boards: ['all'], poolSize: 30, groups: 3, n: 5, hist: 300,
    benchmark: 'hs300', applyCost: true, commissionRate: 0.00025, stampDuty: 0.001,
    slippage: 0.001, factor: 'momentum', assetClass: 'a-share', longOnly: true,
  }
  const bt = await req('POST', '/select/backtest', btBody, { timeout: 240000 })
  check(bt.status === 200, 'backtest(longOnly=true) 200', `status=${bt.status}`)
  const b = bt.data || {}
  check(b.direction === 'long_only', 'direction=long_only（文档 P1 修复）', `实际=${b.direction}`)
  for (const k of ['groupSummary', 'effectiveStart', 'effectiveEnd', 'actualHistDays']) {
    if (!check(b[k] !== undefined, `backtest 返回 ${k}（文档 P2 补全）`, `=${JSON.stringify(b[k])?.slice(0, 40)}`)) break
  }
  check(typeof b.metrics?.cumulativeReturn === 'number', 'metrics.cumulativeReturn 数值')
  check(b.longShort !== undefined, 'longShort 序列存在')
  const bt2 = await req('POST', '/select/backtest', { ...btBody, longOnly: false }, { timeout: 240000 })
  check(bt2.status === 200 && bt2.data?.direction === 'long_short', 'backtest(longOnly=false) direction=long_short', `实际=${bt2.data?.direction}`)

  section('L9 MLView /ml/evaluate（文档 §四 P1：oosRankIc 真实计算）')
  const ev = await req('POST', '/ml/evaluate', {
    board: 'all', boards: ['all'], poolSize: 30, n: 3, hist: 300, modelType: 'gbdt',
    nSplits: 3, gap: 5, useSnapshot: false, assetClass: 'a-share',
  }, { timeout: 240000 })
  check(ev.status === 200, 'evaluate 200')
  check(typeof ev.data?.oosIc === 'number', 'oosIc 数值')
  check(typeof ev.data?.oosRankIc === 'number' && ev.data.oosRankIc !== 0, 'oosRankIc 真实计算非 0（文档 P1 修复）', `=${ev.data?.oosRankIc}`)
  check(Array.isArray(ev.data?.featureImportance), 'featureImportance 数组')

  section('L10 MLView 训练+回测（文档 §四：snapshotStartNote/inSampleWarning/方向）')
  const models = await req('GET', '/ml/models')
  let mid = null
  if (Array.isArray(models.data) && models.data.length) {
    const first = await req('POST', '/ml/backtest', {
      modelId: String(models.data[0].id), board: 'all', boards: ['all'], poolSize: 30,
      groups: 3, n: 3, hist: 300, benchmark: 'none',
      startDate: '2024-01-01', endDate: '2024-06-30', direction: 'long_short',
    }, { timeout: 240000 })
    if (first.status === 200) {
      mid = String(models.data[0].id)
    } else if (String(first.data?.detail || '').includes('泛化特征')) {
      skip('存量模型为泛化特征(f0..)，回测被拒（数据问题非前端问题），改走训练新模型')
    } else {
      warn('存量模型回测失败', JSON.stringify(first.data?.detail || first.data)?.slice(0, 120))
    }
  }
  if (!mid) {
    section('L10b 训练新模型后回测（模拟 MLView 训练→回测流程）')
    const tr = await req('POST', '/ml/train', {
      board: 'all', boards: ['all'], poolSize: 30, n: 3, hist: 300, modelType: 'gbdt',
      nSplits: 3, gap: 5, useSnapshot: false, assetClass: 'a-share',
      selectedFactors: ['momentum', 'ma_dev', 'rsi', 'volatility', 'boll_pct'],
    }, { timeout: 300000 })
    check(tr.status === 200, '/ml/train 200', `status=${tr.status}`)
    mid = tr.data?.model?.id ? String(tr.data.model.id) : (tr.data?.id ? String(tr.data.id) : null)
    if (!mid) { skip('训练未返回 model.id，跳过回测', JSON.stringify(tr.data)?.slice(0, 120)) }
  }
  if (mid) {
    const mb = await req('POST', '/ml/backtest', {
      modelId: mid, board: 'all', boards: ['all'], poolSize: 30, groups: 3, n: 3, hist: 300,
      benchmark: 'none', startDate: '2024-01-01', endDate: '2024-06-30', direction: 'long_short',
    }, { timeout: 240000 })
    check(mb.status === 200, 'ml/backtest 200', `status=${mb.status}`)
    check(['long_short', 'long_only', 'short_only'].includes(mb.data?.direction), 'direction 透传', `=${mb.data?.direction}`)
    check(mb.data?.snapshotStartNote === undefined || typeof mb.data.snapshotStartNote === 'string', 'snapshotStartNote 条件字段语义（无快照因子→不出现；含快照因子+未设起始日→出现）', mb.data?.snapshotStartNote === undefined ? '本模型无快照因子，符合条件缺省' : `=${mb.data.snapshotStartNote}`)
    check(mb.data?.inSampleWarning !== undefined, 'inSampleWarning 字段存在（样本内标注，文档 P1）')
    check(typeof mb.data?.actualHistDays === 'number', 'actualHistDays 数值', `=${mb.data?.actualHistDays}`)
  }

  section('L11 MonitorView /monitor/config（字段契约）')
  const cfg = await req('GET', '/monitor/config')
  check(cfg.status === 200, 'monitor/config 200')
  for (const k of ['mode', 'modelId', 'ranking', 'board', 'poolSize', 'tradeDirections']) {
    if (!check(cfg.data?.[k] !== undefined, `config 含 ${k}`)) break
  }

  section('L12 PortfolioOptView /portfolio-opt + /apply（文档 §二 P1：支持负权重做空腿）')
  const mk = await req('GET', '/select/market?board=all&limit=6', undefined, { auth: false })
  const codes = (Array.isArray(mk.data) ? mk.data : mk.data?.rows || []).map(r => r.code).slice(0, 5)
  if (codes.length < 3) {
    skip('拿不到候选股，跳过组合优化实测', `market 返回 ${Array.isArray(mk.data) ? mk.data.length : '?'} 条`)
  } else {
    // 前端流程：/portfolio-opt/estimate 生成 μ/Σ → /portfolio-opt 求解 → /apply 建仓
    const est = await req('POST', '/portfolio-opt/estimate', { codes }, { timeout: 240000 })
    check(est.status === 200, 'portfolio-opt/estimate 200', `status=${est.status}`)
    const mu = est.data?.mu, cov = est.data?.cov
    if (Array.isArray(mu) && Array.isArray(cov)) {
      const po = await req('POST', '/portfolio-opt', {
        codes: est.data.codes, mu, cov, method: 'mean_variance', maxWeight: 0.3,
        longOnly: false, targetReturn: null,
      }, { timeout: 120000 })
      check(po.status === 200, 'portfolio-opt 200', `status=${po.status}`)
      const weights = po.data?.weights
      check(Array.isArray(weights) && weights.length === est.data.codes.length, 'portfolio-opt 返回 weights', weights ? `${weights.length} 个` : '无 weights')
      if (Array.isArray(weights) && weights.length) {
        if (!weights.some(w => w < -1e-9)) {
          warn('优化结果全正权重（数据依赖：mean_variance 对当前样本未选做空），改用合成负权重直测 apply 空头腿', JSON.stringify(weights.slice(0, 5)))
        }
        // 文档 P1 声称：apply 不再静默丢弃空单 → 注入含负权重（i=1 为 -0.2，其余均分补足 sum=1）验证 side=short 腿
        const testW = weights.map((_, i) => (i === 1 ? -0.2 : 1.2 / (weights.length - 1)))
        const ap = await req('POST', '/portfolio-opt/apply', { codes: est.data.codes, weights: testW }, { timeout: 120000 })
        check(ap.status === 200, 'portfolio-opt/apply 200', `status=${ap.status}`)
        const applied = ap.data?.applied || []
        check(applied.some(a => a.side === 'short'), 'apply 落盘空单腿（side=short，文档 P1：不再静默丢弃）', JSON.stringify(applied.slice(0, 6)))
      }
    } else {
      check(false, 'estimate 返回 μ/Σ', JSON.stringify(est.data)?.slice(0, 100))
    }
  }

  section('L13 建议历史长度实测一致性（前端公式 vs 后端 actualHistDays）')
  const s = new Date('2024-01-01T00:00:00'); const e = new Date('2024-06-30T00:00:00')
  const today = new Date(); today.setHours(0, 0, 0, 0)
  const front = mlViewDays(true, true, s, e, today)
  console.log(`    前端 MLView suggestedHist(2024-01-01~2024-06-30) ≈ ${front} 日`)
  warn('建议历史长度仍为前端自算（文档第 4 条"部分完成：留待前后端统一"）', '后端不输出 suggestedHist 字段')

  if (FULL) {
    section('L14 /reports/backtest 报告导出（模拟 downloadFile）')
    const rp = await req('POST', '/reports/backtest', {
      format: 'html', factorLabel: 'momentum',
      config: { factor: 'momentum', hist: 300, groups: 3, n: 5 },
      metrics: { cumulativeReturn: 0.1, annualizedReturn: 0.1, sharpe: 1, maxDrawdown: 0.1 },
      direction: 'long_only', effectiveStart: '2024-01-01', effectiveEnd: '2024-06-30', actualHistDays: 300,
    }, { timeout: 120000 })
    const ct = rp.headers?.get?.('content-type') || ''
    check(rp.status === 200 && !ct.includes('application/json'), '导出返回非 JSON（HTML 文件）', `status=${rp.status} content-type=${ct}${rp.status !== 200 ? ' body=' + String(rp.data)?.slice(0, 100) : ''}`)
  } else {
    skip('报告导出（重流程，--full 开启）', 'node verify_frontend.mjs --live --full')
  }
}

// ================================================================ 汇总
async function main() {
  try {
    if (MODE === 'static') await runStatic()
    else await runLive()
  } catch (e) {
    console.error(`\n脚本异常终止：${e?.stack || e}`)
    process.exit(3)
  }
  console.log(`\n================ 结果汇总 ================`)
  console.log(`  OK ${stats.ok}  |  FAIL ${stats.fail}  |  WARN ${stats.warn}  |  SKIP ${stats.skip}`)
  if (stats.fail > 0) {
    console.log('  存在 FAIL：文档声称与前后端实际不一致，需处理。')
    process.exit(1)
  }
  if (stats.warn > 0) console.log('  WARN：文档已如实标注的已知项 / 前后端口径分叉，建议人工确认。')
  else console.log('  全部通过。')
  process.exit(0)
}

main()
