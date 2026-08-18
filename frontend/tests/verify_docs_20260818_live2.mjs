// 补测 20260818 两份文档剩余链路（需后端 8001 运行）
// 覆盖：文档2 链路1/2/3/5（规则选股、规则回测、ML评估合法键、ML打分选股）
//       文档1 P2 快照模型回测（snapshotStartNote 正确压缩）+ 训练模型 inSampleWarning 对照
const BASE = process.env.BASE_URL || 'http://127.0.0.1:8001/api'
let token = ''
let ok = 0, fail = 0
function check(name, cond, detail = '') {
  if (cond) { ok++; console.log(`  [PASS] ${name}${detail ? ' — ' + detail : ''}`) }
  else { fail++; console.log(`  [FAIL] ${name}${detail ? ' — ' + detail : ''}`) }
}
async function req(method, path, body, { auth = true, timeout = 300000 } = {}) {
  const headers = {}
  if (body !== undefined) headers['Content-Type'] = 'application/json'
  if (auth && token) headers['Authorization'] = `Bearer ${token}`
  let res
  try {
    res = await fetch(BASE + path, { method, headers, body: body !== undefined ? JSON.stringify(body) : undefined, signal: AbortSignal.timeout(timeout) })
  } catch (e) { return { status: 0, data: String(e?.message || e) } }
  const ct = res.headers.get('content-type') || ''
  let data = null
  try { data = ct.includes('json') ? await res.json() : await res.text() } catch { data = null }
  return { status: res.status, data }
}

async function main() {
  console.log(`BASE_URL = ${BASE}`)
  const uname = 'doc_verify2_' + Date.now().toString(36)
  let login = await req('POST', '/auth/register', { username: uname, password: 'verify_pass_123!' }, { auth: false })
  if (login.status === 400) login = await req('POST', '/auth/login', { username: uname, password: 'verify_pass_123!' }, { auth: false })
  token = login.data?.token || ''
  check('注册/登录', login.status === 200, `status=${login.status}`)
  if (!token) process.exit(1)

  // ===== 文档2 链路1：规则选股（合法键 momentum/volatility/rsi）=====
  console.log('\n=== 文档2 链路1：规则选股（合法键）===')
  const sel = await req('POST', '/select', {
    board: 'all', boards: ['all'], poolSize: 60, topN: 20,
    factors: [
      { key: 'momentum', weight: 1, direction: 1 },
      { key: 'volatility', weight: 0.5, direction: -1 },
      { key: 'rsi', weight: 0.3, direction: -1 },
    ],
    filters: { excludeSt: true }, assetClass: 'a-share',
  }, { timeout: 180000 })
  check('select 200', sel.status === 200, `status=${sel.status} ${JSON.stringify(sel.data?.detail || '')?.slice(0, 80)}`)
  const rows = sel.data?.rows || []
  check('返回候选股 rows>0', rows.length > 0, `rows=${rows.length}`)
  check('带因子 z 分 factorDetail', rows[0]?.factorDetail?.momentum?.raw !== undefined, `raw=${rows[0]?.factorDetail?.momentum?.raw}`)

  // ===== 文档2 链路2：规则因子回测 momentum hist=200 =====
  console.log('\n=== 文档2 链路2：规则因子回测 momentum hist=200 ===')
  const bt = await req('POST', '/select/backtest', {
    board: 'all', boards: ['all'], poolSize: 60, groups: 5, n: 5, hist: 200,
    benchmark: 'none', applyCost: true, factor: 'momentum', assetClass: 'a-share', longOnly: true,
  }, { timeout: 300000 })
  check('backtest 200', bt.status === 200, `status=${bt.status} ${JSON.stringify(bt.data?.detail || '')?.slice(0, 80)}`)
  const m = bt.data?.metrics || {}
  check('rebalanceCount 有限且>0', typeof m.rebalanceCount === 'number' && m.rebalanceCount > 0, `rebalanceCount=${m.rebalanceCount}`)
  const finite = v => typeof v === 'number' && Number.isFinite(v)
  check('cumReturn/sharpe/maxDD 均有限', finite(m.cumulativeReturn) && finite(m.sharpe) && finite(m.maxDrawdown), `cum=${m.cumulativeReturn} sharpe=${m.sharpe} mdd=${m.maxDrawdown}`)

  // ===== 文档2 链路3：ML 评估合法 3 键（特征数=3）=====
  console.log('\n=== 文档2 链路3：ML 评估合法 3 键 momentum/volatility/rsi ===')
  const ev3 = await req('POST', '/ml/evaluate', {
    board: 'all', boards: ['all'], poolSize: 20, n: 3, hist: 300, modelType: 'gbdt',
    nSplits: 3, gap: 5, useSnapshot: false, assetClass: 'a-share',
    selectedFactors: ['momentum', 'volatility', 'rsi'],
  }, { timeout: 240000 })
  check('evaluate 200', ev3.status === 200, `status=${ev3.status}`)
  check('特征数=3（合法键全部保留）', (ev3.data?.featureImportance?.length || 0) === 3, `featureImportance=${JSON.stringify(ev3.data?.featureImportance?.map?.(f => f.feature) || ev3.data?.featureImportance)}`)
  check('无 ignoredFactors（无非法键）', !ev3.data?.ignoredFactors?.length, JSON.stringify(ev3.data?.ignoredFactors))
  check('oosIc/oosRankIc 数值', typeof ev3.data?.oosIc === 'number' && typeof ev3.data?.oosRankIc === 'number', `oosIc=${ev3.data?.oosIc} oosRankIc=${ev3.data?.oosRankIc}`)

  // ===== 文档2 链路5：ML 模型选股 score_latest =====
  console.log('\n=== 文档2 链路5：ML 模型选股 score_latest（人造模型）===')
  const sc = await req('POST', '/ml/score', {
    modelId: 'manual_20260818_164659', board: 'all', boards: ['all'], poolSize: 100,
  }, { timeout: 240000 })
  check('score 200', sc.status === 200, `status=${sc.status} ${JSON.stringify(sc.data?.detail || '')?.slice(0, 80)}`)
  const longList = sc.data?.longList || sc.data?.scores || []
  check('返回打分排序股>0', longList.length > 0, `len=${longList.length}`)
  const s0 = Array.isArray(longList) ? longList[0] : null
  check('含 code+score', s0 && typeof s0.code === 'string' && typeof s0.score === 'number', JSON.stringify(s0)?.slice(0, 80))

  // ===== 文档1 P2：含快照因子的人造模型 → 正确压缩 + snapshotStartNote =====
  console.log('\n=== 文档1 P2：含快照因子模型回测（正确压缩行为）===')
  const mk = await req('POST', '/ml/models/manual', {
    name: 'snap_verify', featureWeights: { momentum: 0.5, pe: 0.5 }, threshold: 0,
    rule: '', bullRule: '', bearRule: '', direction: 'long_short', allowShort: true,
  })
  check('创建含快照 manual 模型', mk.status === 200, `status=${mk.status} ${JSON.stringify(mk.data?.detail || '')?.slice(0, 60)}`)
  const snapId = mk.data?.id
  check('featureNames 仅 momentum+pe', snapId && (mk.data?.featureNames || []).join(',') === 'momentum,pe', JSON.stringify(mk.data?.featureNames))
  if (snapId) {
    const sb = await req('POST', '/ml/backtest', {
      modelId: snapId, board: 'all', boards: ['all'], poolSize: 30, groups: 3, n: 3,
      hist: 300, benchmark: 'none', direction: 'long_short',
    }, { timeout: 300000 })
    check('含快照模型回测 200', sb.status === 200, `status=${sb.status}`)
    check('出现 snapshotStartNote（含快照因子+未设起始日）', typeof sb.data?.snapshotStartNote === 'string', sb.data?.snapshotStartNote?.slice(0, 60))
    check('无 inSampleWarning（manual）', sb.data?.inSampleWarning === undefined)
    const rbc = sb.data?.metrics?.rebalanceCount
    check('rebalanceCount 被正确压缩（≤ 最近60日/n）', typeof rbc === 'number' && rbc > 0 && rbc <= 25, `rebalanceCount=${rbc}`)
  }

  // ===== 文档1 P2 对照 + 文档2 链路6：训练真实因子模型 → inSampleWarning 应出现 =====
  console.log('\n=== 训练真实因子模型 → 回测 inSampleWarning 对照（慢）===')
  const tr = await req('POST', '/ml/train', {
    board: 'all', boards: ['all'], poolSize: 20, n: 3, hist: 300, modelType: 'gbdt',
    nSplits: 3, gap: 5, useSnapshot: false, assetClass: 'a-share',
    selectedFactors: ['momentum', 'volatility', 'rsi'],
  }, { timeout: 420000 })
  check('train 200', tr.status === 200, `status=${tr.status} ${JSON.stringify(tr.data?.detail || '')?.slice(0, 80)}`)
  const mid = tr.data?.model?.id || tr.data?.id
  if (mid) {
    const tb = await req('POST', '/ml/backtest', {
      modelId: String(mid), board: 'all', boards: ['all'], poolSize: 30, groups: 3, n: 3,
      hist: 200, benchmark: 'none', direction: 'long_short',
    }, { timeout: 300000 })
    check('训练模型回测 200', tb.status === 200, `status=${tb.status} ${JSON.stringify(tb.data?.detail || '')?.slice(0, 80)}`)
    check('训练模型有 inSampleWarning（对照 manual 无）', typeof tb.data?.inSampleWarning === 'string', tb.data?.inSampleWarning?.slice(0, 50))
    const rbc2 = tb.data?.metrics?.rebalanceCount
    check('训练模型跑满历史（rebalanceCount>20）', typeof rbc2 === 'number' && rbc2 > 20, `rebalanceCount=${rbc2}`)
  }

  console.log(`\n结果: ${ok} 通过, ${fail} 失败`)
  process.exit(fail ? 1 : 0)
}
main().catch(e => { console.error('脚本异常:', e); process.exit(2) })
