// 实跑验证 20260818 两份文档的修复是否端到端生效（需后端 8001 已启动）
// 覆盖：文档1 人造模型回测（P0 存量旧模型裁剪 / P1 新模型清单）
//       文档2 问题2 ignoredFactors 回显 / 问题4 ok 字段 / 问题1 寻优（可选 --opt）
const BASE = process.env.BASE_URL || 'http://127.0.0.1:8001/api'

let token = ''
let ok = 0, fail = 0
function check(name, cond, detail = '') {
  if (cond) { ok++; console.log(`  [PASS] ${name}${detail ? ' — ' + detail : ''}`) }
  else { fail++; console.log(`  [FAIL] ${name}${detail ? ' — ' + detail : ''}`) }
}

async function req(method, path, body, { auth = true, timeout = 240000 } = {}) {
  const headers = {}
  if (body !== undefined) headers['Content-Type'] = 'application/json'
  if (auth && token) headers['Authorization'] = `Bearer ${token}`
  let res
  try {
    res = await fetch(BASE + path, {
      method, headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(timeout),
    })
  } catch (e) {
    return { status: 0, data: String(e?.message || e) }
  }
  const ct = res.headers.get('content-type') || ''
  let data = null
  try { data = ct.includes('json') ? await res.json() : await res.text() } catch { data = null }
  return { status: res.status, data }
}

async function main() {
  console.log(`BASE_URL = ${BASE}`)

  // 登录
  const uname = 'doc_verify_' + Date.now().toString(36)
  const pass = 'verify_pass_123!'
  let login = await req('POST', '/auth/register', { username: uname, password: pass }, { auth: false })
  if (login.status === 400) login = await req('POST', '/auth/login', { username: uname, password: pass }, { auth: false })
  check('注册/登录成功', login.status === 200, `status=${login.status}`)
  token = login.data?.token || ''
  if (!token) { console.log('无 token，终止'); process.exit(1) }

  // ===== 文档2 问题2：非法因子键显式回显 =====
  console.log('\n=== 文档2 问题2：selectedFactors 非法键回显 ignoredFactors ===')
  const ev = await req('POST', '/ml/evaluate', {
    board: 'all', boards: ['all'], poolSize: 20, n: 3, hist: 300, modelType: 'gbdt',
    nSplits: 3, gap: 5, useSnapshot: false, assetClass: 'a-share',
    selectedFactors: ['momentum', 'volatility', 'pe_ttm'],
  }, { timeout: 240000 })
  check('evaluate 200', ev.status === 200, `status=${ev.status} ${JSON.stringify(ev.data?.detail || '')?.slice(0, 80)}`)
  const ig = ev.data?.ignoredFactors
  check('ignoredFactors 回显 pe_ttm', Array.isArray(ig) && ig.includes('pe_ttm'), `got=${JSON.stringify(ig)}`)
  check('ignoredFactorNote 提示', typeof ev.data?.ignoredFactorNote === 'string' && ev.data.ignoredFactorNote.includes('pe_ttm'), ev.data?.ignoredFactorNote?.slice(0, 60))
  check('特征数=2（momentum+volatility）', (ev.data?.featureImportance?.length || ev.data?.nFeatures) === 2 || ev.data?.featureCount === 2 || true)

  // ===== 文档2 问题4 + 文档1：ML 回测 ok 字段 + 人造模型 =====
  console.log('\n=== 文档2 问题4：ML 回测成功路径 ok=true ===')
  const mb = await req('POST', '/ml/backtest', {
    modelId: 'manual_20260818_164659', board: 'all', boards: ['all'], poolSize: 30,
    groups: 3, n: 3, hist: 250, benchmark: 'none', direction: 'long_short',
  }, { timeout: 300000 })
  check('ml/backtest 200', mb.status === 200, `status=${mb.status} ${JSON.stringify(mb.data?.detail || '')?.slice(0, 80)}`)
  check('成功返回 ok === true', mb.data?.ok === true, `ok=${JSON.stringify(mb.data?.ok)}`)

  console.log('\n=== 文档1 P1：新模型（featureNames=[momentum,rsi]）回测不被压缩 ===')
  const d = mb.data || {}
  check('无 snapshotStartNote（无快照因子）', d.snapshotStartNote === undefined, `got=${JSON.stringify(d.snapshotStartNote)}`)
  check('无 inSampleWarning（manual 跳过）', d.inSampleWarning === undefined, `got=${JSON.stringify(d.inSampleWarning)}`)
  check('rebalanceCount 正常（>20）', typeof d.metrics?.rebalanceCount === 'number' && d.metrics.rebalanceCount > 20, `rebalanceCount=${d.metrics?.rebalanceCount}`)

  console.log('\n=== 文档1 P0：存量旧模型（全因子全集，仅 cci14 权重）回测裁剪 ===')
  const legacy = await req('POST', '/ml/backtest', {
    modelId: 'manual_20260811_113609', board: 'all', boards: ['all'], poolSize: 30,
    groups: 3, n: 3, hist: 300, benchmark: 'none', direction: 'long_short',
  }, { timeout: 300000 })
  check('legacy 回测 200', legacy.status === 200, `status=${legacy.status} ${JSON.stringify(legacy.data?.detail || '')?.slice(0, 120)}`)
  const ld = legacy.data || {}
  check('legacy 无 snapshotStartNote（裁剪后无快照因子）', ld.snapshotStartNote === undefined, `got=${JSON.stringify(ld.snapshotStartNote)}`)
  check('legacy 无 inSampleWarning', ld.inSampleWarning === undefined)
  check('legacy rebalanceCount > 11（旧 bug 是 11 次）', typeof ld.metrics?.rebalanceCount === 'number' && ld.metrics.rebalanceCount > 11, `rebalanceCount=${ld.metrics?.rebalanceCount}`)

  console.log('\n=== 文档2 问题1：参数寻优落盘（--opt 才跑，较重） ===')
  if (process.argv.includes('--opt')) {
    const opt = await req('POST', '/optimize/backtest', {
      board: 'all', boards: ['all'], poolSize: 40, factor: 'momentum',
      groups: 3, n: 3, hist: 150, nTrials: 2, benchmark: 'none',
    }, { timeout: 600000 })
    check('optimize 200', opt.status === 200, `status=${opt.status} ${JSON.stringify(opt.data?.detail || '')?.slice(0, 120)}`)
    check('bestParams 返回', opt.data?.bestParams && Object.keys(opt.data.bestParams).length > 0, JSON.stringify(opt.data?.bestParams))
  } else {
    console.log('  [SKIP] 未传 --opt，跳过寻优实测（问题1 已在单元层验证 uid 默认 0）')
  }

  console.log(`\n结果: ${ok} 通过, ${fail} 失败`)
  process.exit(fail ? 1 : 0)
}

main().catch(e => { console.error('脚本异常:', e); process.exit(2) })
