<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import api, { longTask, downloadFile, downloadGet } from '../api/client'
import { useToast } from '../stores/toast'
import { useResearchStore } from '../stores/research'
import EChart from '../components/EChart.vue'
import Skeleton from '../components/Skeleton.vue'
import BoardSelect from '../components/BoardSelect.vue'

const { toast } = useToast()
const research = useResearchStore()

const BOARD_OPTIONS = [
  { value: 'all', label: '全部A股' },
  { value: 'sh_main', label: '沪市主板' },
  { value: 'sz_main', label: '深市主板' },
  { value: 'gem', label: '创业板' },
  { value: 'star', label: '科创板' },
  { value: 'bse', label: '北交所' }
]
const BENCH_OPTIONS = [
  { value: 'none', label: '无基准' },
  { value: 'hs300', label: '沪深300' },
  { value: 'zz500', label: '中证500' },
  { value: 'sse', label: '上证指数' }
]

const boards = ref(['all'])
const poolSize = ref(60)
const factorKey = ref('momentum')
const strategySource = ref('factor')   // factor=技术因子 | model=ML模型
const modelId = ref('')
const groups = ref(5)
const days = ref(5)
const hist = ref(300)
const benchmark = ref('hs300')
const applyCost = ref(true)
const commissionRate = ref(0.00025)
const stampDuty = ref(0.001)
const slippage = ref(0.001)
// P6：回测时间区间（后端 BacktestBody 已支持，前端补控件）
const startDate = ref('')
const endDate = ref('')
// P7：行业板块（board 传 "sector:<行业名>"）
const sectorOptions = ref([])
// P2：资产类别（future 时候选池取期货主力连续合约）
const assetClass = ref('a-share')

// 起止日联动 hist：hist 是拉取 K 线的数据窗口天数（从今天往前回溯），给因子计算+暖机用。
// 覆盖条件：窗口须回溯到 startDate 之前（暖机 60 + 长周期因子 240）；
// 双端场景只按 startDate 回溯即可（今天往前覆盖 startDate 天然包含 endDate）；
// 仅 endDate 时窗口结束于 endDate，需额外覆盖 (today - endDate) 区间。
watch([startDate, endDate], ([s, e]) => {
  if (s || e) {
    const today = new Date(); today.setHours(0, 0, 0, 0)
    const from = s ? new Date(s + 'T00:00:00') : null
    const to = e ? new Date(e + 'T00:00:00') : null
    let need = 300
    if (from) {
      need = Math.floor((today - from) / 86400000) + 60 + 240
    } else if (to && to < today) {
      need = Math.floor((today - to) / 86400000) + 60 + 240
    }
    hist.value = Math.max(300, Math.ceil(need))
  } else {
    hist.value = 300
  }
})

async function loadSectors() {
  try {
    const map = await api.get('/data/sectors')
    const names = [...new Set(Object.values(map || {}))].filter(Boolean).sort((a, b) => a.localeCompare(b, 'zh'))
    sectorOptions.value = names.map(n => ({ value: 'sector:' + n, label: n }))
  } catch (e) { /* 行业数据拉取失败不阻塞页面 */ }
}

const loading = ref(false)
const saving = ref(false)
const exporting = ref(false)
const result = ref(null)
const factorOptions = ref([])
const modelOptions = ref([])
const showSave = ref(false)
const strategyName = ref('')

// 从中间结果（选股 artifact）读取自定义股池，回测直接使用该 codes
const artifactInput = ref('')
const artifactCodes = ref([])
const loadingArtifact = ref(false)
async function loadArtifact() {
  const aid = artifactInput.value.trim()
  if (!aid) { toast('请输入中间结果 ID'); return }
  loadingArtifact.value = true
  try {
    const rec = await api.get('/artifacts/' + aid)
    const codes = rec.payload?.codes || []
    if (!codes.length) { toast('该中间结果不含股票代码'); return }
    artifactCodes.value = codes
    toast(`已载入 ${codes.length} 只股票（${rec.name || ''}）`)
  } catch (e) { toast(e.message) }
  finally { loadingArtifact.value = false }
}

async function loadFactorOptions() {
  const data = await api.get('/select/factors')
  factorOptions.value = data.filter(f => f.kline)
}

async function loadModels() {
  try { modelOptions.value = await api.get('/ml/models') }
  catch (e) { /* 静默：模型列表拉取失败不阻塞页面 */ }
}

const backtestConfig = () => {
  const cfg = {
    board: 'all', boards: boards.value, poolSize: Number(poolSize.value),
    groups: Number(groups.value), n: Number(days.value), hist: Number(hist.value),
    benchmark: benchmark.value, applyCost: applyCost.value,
    commissionRate: Number(commissionRate.value), stampDuty: Number(stampDuty.value),
    slippage: Number(slippage.value),
    startDate: startDate.value || null, endDate: endDate.value || null,
    assetClass: assetClass.value,
  }
  if (artifactCodes.value.length) cfg.codes = artifactCodes.value
  if (strategySource.value === 'model') {
    cfg.modelId = modelId.value
  } else {
    cfg.factor = factorKey.value
  }
  return cfg
}

async function runBacktest() {
  if (strategySource.value === 'model' && !modelId.value) {
    toast('请先选择 ML 模型'); return
  }
  // 表单校验：清空输入框时 Number('')=0 会发出脏请求（poolSize=0/groups=0/n=0）
  const ps = Number(poolSize.value), gp = Number(groups.value), dn = Number(days.value), hs = Number(hist.value)
  if (!ps || ps < 20) { toast('候选池规模需 ≥ 20'); return }
  if (!gp || gp < 2) { toast('分组数需 ≥ 2'); return }
  if (!dn || dn < 1) { toast('持有天数需 ≥ 1'); return }
  if (!hs || hs < 60) { toast('历史长度需 ≥ 60'); return }
  // 前置校验：groups×3 约束（最常见 422 根因，提前拦截减少无效请求）
  if (artifactCodes.value.length > 0) {
    if (artifactCodes.value.length < gp * 3) {
      toast(`自选仅 ${artifactCodes.value.length} 只，但分组 ${gp} 有效样本需 ≥ ${gp * 3} 只（每组至少 3 只才有统计意义）。请减少分组数至 ≤ ${Math.floor(artifactCodes.value.length / 3)} 或增加自选股票。`)
      return
    }
  } else {
    if (ps < gp * 3) {
      toast(`候选池 ${ps} 只 < 分组 ${gp} 所需 ${gp * 3} 只（groups×3）。请减少分组数 ≤ ${Math.floor(ps / 3)} 或扩大候选池。`)
      return
    }
  }
  loading.value = true
  try {
    result.value = await longTask('/select/backtest', backtestConfig())
  } catch (e) {
    const msg = e.message || ''
    if (msg.includes('422') || msg.includes('样本不足')) {
      toast(`回测失败：${msg}\n提示：减少分组数、扩大候选池或错峰重试。`)
    } else if (msg.includes('502') || msg.includes('限流')) {
      toast(`回测失败：行情上游限流。\n建议：切换到「全部A股」板块、减小候选池规模、或错峰（避开开盘/收盘高峰）重试。`)
    } else {
      toast(msg)
    }
    result.value = null
  } finally { loading.value = false }
}

async function saveStrategy() {
  if (!strategyName.value.trim()) { toast('请输入策略名称'); return }
  saving.value = true
  try {
    await api.post('/strategies', { name: strategyName.value.trim(), kind: 'backtest', config: backtestConfig() })
    toast('策略已保存'); showSave.value = false; strategyName.value = ''
  } catch (e) { toast(e.message) }
  finally { saving.value = false }
}

async function exportReport(fmt) {
  if (!result.value) return
  exporting.value = true
  try {
    const payload = {
      format: fmt, factorLabel: result.value.factorLabel, config: result.value.config,
      metrics: result.value.metrics, benchmark: result.value.benchmark,
      groupSummary: result.value.groupSummary, longShort: result.value.longShort,
      icSeries: result.value.icSeries,
      positionLedger: result.value.positionLedger || [],
      featureImportance: result.value.featureImportance || [],
      actualHistDays: result.value.actualHistDays,
      effectiveStart: result.value.effectiveStart,
      effectiveEnd: result.value.effectiveEnd,
      direction: result.value.direction || 'long_short',
      survivorshipBiasWarning: result.value.survivorshipBiasWarning || '',
      snapshotWarning: result.value.snapshotWarning || '',
      histWarning: result.value.histWarning || '',
      icIr: result.value.icIr ?? null,
      icTStat: result.value.icTStat ?? null,
      icPValue: result.value.icPValue ?? null,
      yearlyReturns: result.value.yearlyReturns || [],
      stockContribution: result.value.stockContribution || [],
      bucketDates: result.value.bucketDates || [],
    }
    const ext = fmt === 'html' ? 'html' : fmt === 'pdf' ? 'pdf' : 'xlsx'
    await downloadFile('/reports/backtest', payload, `backtest.${ext}`)
  } catch (e) { toast(e.message) }
  finally { exporting.value = false }
}

// P10：报告历史（回测完成自动存档，可下载/删除）
const reportRuns = ref([])
const loadingRuns = ref(false)
async function loadReportRuns() {
  loadingRuns.value = true
  try { reportRuns.value = await api.get('/reports/runs?limit=50') }
  catch (e) { /* 静默 */ }
  finally { loadingRuns.value = false }
}
async function downloadRun(run, fmt) {
  try {
    if (fmt === 'html') {
      await downloadGet(`/reports/runs/${run.id}`, `report_${run.id}.html`)
    } else {
      await downloadFile(`/reports/runs/${run.id}/regenerate?fmt=${fmt}`, {}, `report_${run.id}.${fmt}`)
    }
  } catch (e) { toast(e.message) }
}
async function deleteRun(run) {
  if (!confirm(`删除报告记录 #${run.id}？`)) return
  try {
    await api.delete(`/reports/runs/${run.id}`)
    reportRuns.value = reportRuns.value.filter(r => r.id !== run.id)
  } catch (e) { toast(e.message) }
}

const m = computed(() => result.value?.metrics || {})

const groupBarOption = computed(() => {
  if (!result.value) return {}
  const data = result.value.groupSummary
  return {
    backgroundColor: 'transparent',
    grid: { left: 60, right: 30, top: 30, bottom: 40 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: data.map(d => `G${d.group}`), axisLine: { lineStyle: { color: '#d7dce8' } }, axisLabel: { color: '#8a94a6' } },
    yAxis: { type: 'value', name: `平均未来${result.value.n}日收益`, nameTextStyle: { color: '#8a94a6' }, axisLine: { lineStyle: { color: '#d7dce8' } }, axisLabel: { color: '#8a94a6', formatter: v => (v * 100).toFixed(1) + '%' }, splitLine: { lineStyle: { color: '#e9edf5' } } },
    series: [{ type: 'bar', data: data.map(d => d.avgReturn), itemStyle: { color: p => (p.value >= 0 ? '#ff4d4f' : '#21c08b') } }]
  }
})

const icLineOption = computed(() => {
  if (!result.value) return {}
  const s = result.value.icSeries
  return {
    backgroundColor: 'transparent',
    grid: { left: 60, right: 30, top: 40, bottom: 60 },
    tooltip: { trigger: 'axis' },
    legend: { textStyle: { color: '#5b6675' }, top: 0 },
    xAxis: { type: 'category', data: s.map(d => d.date), axisLine: { lineStyle: { color: '#d7dce8' } }, axisLabel: { color: '#8a94a6', rotate: 45 } },
    yAxis: { type: 'value', axisLine: { lineStyle: { color: '#d7dce8' } }, axisLabel: { color: '#8a94a6' }, splitLine: { lineStyle: { color: '#e9edf5' } } },
    series: [
      { name: 'IC', type: 'line', data: s.map(d => d.ic), showSymbol: false, lineStyle: { color: '#4f8cff', width: 2 } },
      { name: 'RankIC', type: 'line', data: s.map(d => d.rankIc), showSymbol: false, lineStyle: { color: '#6c5ce7', width: 2 } }
    ]
  }
})

const longShortOption = computed(() => {
  if (!result.value) return {}
  const s = result.value.longShort
  const series = [{ name: '多空累计', type: 'line', data: s.map(d => d.cum), showSymbol: false, lineStyle: { color: '#ff4d4f', width: 2 }, areaStyle: { color: 'rgba(255,77,79,.12)' } }]
  if (result.value.benchmark) {
    series.push({ name: '最高组累计', type: 'line', data: s.map(d => d.topCum), showSymbol: false, lineStyle: { color: '#21c08b', width: 2 } })
  }
  return {
    backgroundColor: 'transparent',
    grid: { left: 60, right: 30, top: 40, bottom: 60 },
    tooltip: { trigger: 'axis' },
    legend: { textStyle: { color: '#5b6675' }, top: 0 },
    xAxis: { type: 'category', data: s.map(d => d.date), axisLine: { lineStyle: { color: '#d7dce8' } }, axisLabel: { color: '#8a94a6', rotate: 45 } },
    yAxis: { type: 'value', axisLine: { lineStyle: { color: '#d7dce8' } }, axisLabel: { color: '#8a94a6', formatter: v => (v * 100).toFixed(0) + '%' }, splitLine: { lineStyle: { color: '#e9edf5' } } },
    series
  }
})

const positionOption = computed(() => {
  if (!result.value) return {}
  const pl = result.value.positionLedger || []
  if (!pl.length) return {}
  const longCount = pl.map(p => p.longCount !== undefined ? p.longCount : (p.long ? p.long.length : 0))
  const shortCount = pl.map(p => p.shortCount !== undefined ? p.shortCount : (p.short ? p.short.length : 0))
  const net = pl.map((p, i) => longCount[i] - shortCount[i])
  const turnover = pl.map((p, i) => {
    if (p.turnover !== undefined) return p.turnover
    if (i === 0) return longCount[0] + shortCount[0]
    const prev = new Set([...(pl[i - 1].long || []), ...(pl[i - 1].short || [])])
    const curr = new Set([...(p.long || []), ...(p.short || [])])
    let same = 0; prev.forEach(c => { if (curr.has(c)) same++ })
    return prev.size + curr.size - 2 * same
  })
  return {
    backgroundColor: 'transparent',
    grid: { left: 60, right: 50, top: 40, bottom: 60 },
    tooltip: { trigger: 'axis' },
    legend: { textStyle: { color: '#5b6675' }, top: 0 },
    xAxis: { type: 'category', data: pl.map(p => p.date), axisLine: { lineStyle: { color: '#d7dce8' } }, axisLabel: { color: '#8a94a6', rotate: 45 } },
    yAxis: [
      { type: 'value', name: '持仓数', nameTextStyle: { color: '#8a94a6' }, axisLine: { lineStyle: { color: '#d7dce8' } }, axisLabel: { color: '#8a94a6' }, splitLine: { lineStyle: { color: '#e9edf5' } } },
      { type: 'value', name: '换手数', nameTextStyle: { color: '#8a94a6' }, axisLine: { lineStyle: { color: '#d7dce8' } }, axisLabel: { color: '#8a94a6' }, splitLine: { show: false } }
    ],
    series: [
      { name: '多头持仓', type: 'line', data: longCount, showSymbol: false, lineStyle: { color: '#ff4d4f', width: 2 } },
      { name: '空头持仓', type: 'line', data: shortCount, showSymbol: false, lineStyle: { color: '#21c08b', width: 2 } },
      { name: '净持仓(多-空)', type: 'line', data: net, showSymbol: false, lineStyle: { color: '#4f8cff', width: 2, type: 'dashed' } },
      { name: '换手股数', type: 'bar', yAxisIndex: 1, data: turnover, itemStyle: { color: 'rgba(79,140,255,.25)' } }
    ]
  }
})

const pct = v => v == null ? '-' : (v * 100).toFixed(2) + '%'
const num = (v, d = 4) => v == null ? '-' : Number(v).toFixed(d)

onMounted(async () => {
  await Promise.all([loadFactorOptions(), loadModels(), loadSectors()])
  // 消费 ML 页写入的当前模型：有则自动切到模型来源并选中（打通 ML→主回测闭环）
  const cm = research.currentModel
  if (cm?.id && modelOptions.value.some(m => m.id === cm.id)) {
    strategySource.value = 'model'
    modelId.value = cm.id
    toast('已预填当前 ML 模型，可直接回测')
  }
  // 消费寻优页回填的最优参数（researchStore 跨页共享）
  const p = research.consumeOptimalParams()
  if (p) {
    if (p.boards) boards.value = p.boards
    else if (p.board) boards.value = [p.board]
    if (p.poolSize) poolSize.value = p.poolSize
    if (p.groups) groups.value = p.groups
    if (p.n) days.value = p.n
    if (p.hist) hist.value = p.hist
    if (p.benchmark) benchmark.value = p.benchmark
    if (p.modelId) {
      strategySource.value = 'model'
      modelId.value = p.modelId
    } else if (p.factor) {
      strategySource.value = 'factor'
      factorKey.value = p.factor
    }
    toast('已预填寻优最优参数，可直接回测')
  }
})
</script>

<template>
  <div>
    <div class="panel-toolbar">
      <div class="field"><label>资产类别</label>
        <select v-model="assetClass">
          <option value="a-share">A股</option>
          <option value="future">期货（主力连续）</option>
        </select>
      </div>
      <BoardSelect v-model="boards" />
      <div class="field"><label>候选池规模</label><input v-model="poolSize" type="number" min="20" max="300" /></div>
      <div class="field"><label>策略来源</label>
        <select v-model="strategySource">
          <option value="factor">技术因子</option>
          <option value="model">ML模型</option>
        </select>
      </div>
      <div v-if="strategySource==='factor'" class="field"><label>因子</label>
        <select v-model="factorKey"><option v-for="f in factorOptions" :key="f.key" :value="f.key">{{ f.label }}</option></select>
      </div>
      <div v-else class="field"><label>模型</label>
        <select v-model="modelId">
          <option value="">请选择模型</option>
          <option v-for="m in modelOptions" :key="m.id" :value="m.id">{{ m.id }}</option>
        </select>
      </div>
      <div class="field"><label>分组数</label><input v-model="groups" type="number" min="2" max="10" /></div>
      <div class="field"><label>持有天数 N</label><input v-model="days" type="number" min="1" max="30" /></div>
      <div class="field"><label>历史长度(日)</label><input v-model="hist" type="number" min="60" max="5000" /></div>
      <div class="field"><label>回测起始日</label><input v-model="startDate" type="date" /></div>
      <div class="field"><label>回测结束日</label><input v-model="endDate" type="date" /></div>
      <div class="field"><label>基准</label>
        <select v-model="benchmark"><option v-for="b in BENCH_OPTIONS" :key="b.value" :value="b.value">{{ b.label }}</option></select>
      </div>
      <button class="btn-primary" :disabled="loading" @click="runBacktest">{{ loading ? '回测中…' : '运行分层回测' }}</button>
    </div>
    <div class="panel-toolbar" style="margin-top:8px">
      <div class="field grow"><label>中间结果 ID（选股页保存的 artifact）</label><input v-model="artifactInput" placeholder="留空则用上方板块/规模拉取候选池" /></div>
      <button class="btn-ghost" :disabled="loadingArtifact" @click="loadArtifact">{{ loadingArtifact ? '载入中…' : '载入股池' }}</button>
      <span v-if="artifactCodes.length" class="hint">已用 {{ artifactCodes.length }} 只自定义股池（{{ artifactCodes.slice(0,3).join(',') }}…）</span>
    </div>
    <div class="panel-toolbar" style="margin-top:8px">
      <div class="field"><label><input type="checkbox" v-model="applyCost" /> 启用交易成本</label></div>
      <div class="field"><label>佣金(万)</label><input v-model="commissionRate" type="number" step="0.00001" :disabled="!applyCost" /></div>
      <div class="field"><label>印花税(千)</label><input v-model="stampDuty" type="number" step="0.0001" :disabled="!applyCost" /></div>
      <div class="field"><label>滑点</label><input v-model="slippage" type="number" step="0.0001" :disabled="!applyCost" /></div>
      <button class="btn-ghost" @click="showSave = !showSave" :disabled="!result">保存为策略</button>
      <button class="btn-ghost" @click="exportReport('html')" :disabled="!result || exporting">导出HTML</button>
      <button class="btn-ghost" @click="exportReport('excel')" :disabled="!result || exporting">导出Excel</button>
      <button class="btn-ghost" @click="exportReport('pdf')" :disabled="!result || exporting">导出PDF</button>
      <button class="btn-ghost" @click="loadReportRuns" :disabled="loadingRuns">{{ loadingRuns ? '加载中…' : '报告历史' }}</button>
    </div>
    <div v-if="reportRuns.length" class="card" style="margin-top:12px">
      <div class="card-head"><h3>报告历史（回测完成自动存档）</h3><span class="hint">共 {{ reportRuns.length }} 条</span></div>
      <table class="data-table">
        <thead><tr><th>ID</th><th>创建时间</th><th>累计收益</th><th>Sharpe</th><th>报告文件</th><th>操作</th></tr></thead>
        <tbody>
          <tr v-for="r in reportRuns" :key="r.id">
            <td>{{ r.id }}</td>
            <td>{{ (r.created_at || '').slice(0, 19).replace('T', ' ') }}</td>
            <td>{{ pct(r.metrics?.cumulativeReturn) }}</td>
            <td>{{ num(r.metrics?.sharpe) }}</td>
            <td class="muted">{{ r.report_path || '已删除' }}</td>
            <td>
              <button class="btn-ghost sm" @click="downloadRun(r, 'html')">HTML</button>
              <button class="btn-ghost sm" @click="downloadRun(r, 'excel')">Excel</button>
              <button class="btn-ghost sm" @click="downloadRun(r, 'pdf')">PDF</button>
              <button class="btn-ghost sm danger" @click="deleteRun(r)">删除</button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    <div v-if="showSave" class="panel-toolbar" style="margin-top:8px">
      <div class="field"><label>策略名称</label><input v-model="strategyName" placeholder="如：动量分层_v1" /></div>
      <button class="btn-primary" :disabled="saving" @click="saveStrategy">{{ saving ? '保存中…' : '确认保存' }}</button>
    </div>

    <div v-if="loading" class="stat-cards"><Skeleton type="cards" :count="8" /></div>
    <div v-else-if="!result" class="empty-hint">设置参数后点击「运行分层回测」</div>
    <template v-else>
      <div v-if="result.histWarning" class="warn-box" style="margin-bottom:12px">
        📌 {{ result.histWarning }}
      </div>
      <div v-if="result.survivorshipBiasWarning" class="warn-box" style="margin-bottom:12px">
        ⚠️ {{ result.survivorshipBiasWarning }}
      </div>
      <div v-if="result.snapshotWarning" class="warn-box" style="margin-bottom:12px;background:rgba(230,168,23,.12);border-color:#e6a81755;color:#e6a817">
        ⚠️ {{ result.snapshotWarning }}
      </div>
      <div class="stat-cards">
        <div class="stat-card"><div class="label">有效股票数</div><div class="value">{{ result.effectiveStocks }}</div></div>
        <div class="stat-card"><div class="label">调仓期数</div><div class="value">{{ result.rebalanceCount }}</div></div>
        <div class="stat-card"><div class="label">有效区间</div><div class="value hint" style="font-size:13px">{{ result.effectiveStart || '-' }} ~ {{ result.effectiveEnd || '-' }}</div></div>
        <div class="stat-card"><div class="label">实际K线天数</div><div class="value">{{ result.actualHistDays || '-' }}</div></div>
        <div class="stat-card"><div class="label">平均 IC</div><div class="value" :class="result.meanIc >= 0 ? 'up' : 'down'">{{ num(result.meanIc) }}</div></div>
        <div class="stat-card"><div class="label">IC 胜率</div><div class="value">{{ (result.icWinRate * 100).toFixed(1) }}%</div></div>
        <div class="stat-card"><div class="label">年化收益</div><div class="value up">{{ pct(m.annualizedReturn) }}</div></div>
        <div class="stat-card"><div class="label">Sharpe</div><div class="value">{{ num(m.sharpe) }}</div></div>
        <div class="stat-card"><div class="label">最大回撤</div><div class="value down">{{ pct(m.maxDrawdown) }}</div></div>
        <div class="stat-card"><div class="label">胜率</div><div class="value">{{ pct(m.winRate) }}</div></div>
      </div>

      <div v-if="result.benchmark" class="stat-cards">
        <div class="stat-card"><div class="label">基准累计</div><div class="value">{{ pct(result.benchmark.cumulativeReturn) }}</div></div>
        <div class="stat-card"><div class="label">基准Sharpe</div><div class="value">{{ num(result.benchmark.sharpe) }}</div></div>
        <div class="stat-card"><div class="label">Alpha</div><div class="value">{{ num(result.benchmark.alpha) }}</div></div>
        <div class="stat-card"><div class="label">Beta</div><div class="value">{{ num(result.benchmark.beta) }}</div></div>
        <div class="stat-card"><div class="label">Calmar</div><div class="value">{{ num(m.calmar) }}</div></div>
        <div class="stat-card"><div class="label">Sortino</div><div class="value">{{ num(m.sortino) }}</div></div>
      </div>

      <div class="card chart-card"><div class="card-head"><h3>分层收益（组1=因子值最低，组N=因子值最高）</h3></div><EChart :option="groupBarOption" height="320px" /></div>
      <div class="card chart-card"><div class="card-head"><h3>IC / RankIC 时序列</h3></div><EChart :option="icLineOption" height="320px" /></div>
      <div class="card chart-card"><div class="card-head"><h3>多空组合累计收益曲线</h3></div><EChart :option="longShortOption" height="320px" /></div>
      <div class="card chart-card" v-if="(result.positionLedger || []).length"><div class="card-head"><h3>仓位变化过程</h3><span class="hint">多头/空头持仓数、净持仓与逐期换手</span></div><EChart :option="positionOption" height="320px" /></div>
    </template>
  </div>
</template>

<style scoped>
.data-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
.data-table th, .data-table td { border: 1px solid var(--border); padding: 9px 11px; text-align: left; }
.data-table th { background: var(--card-2); color: var(--text-dim); font-weight: 600; }
.btn-ghost.sm { padding: 4px 10px; font-size: 12px; margin-right: 4px; }
.btn-ghost.sm.danger { color: #ff6b6b; }
.muted { color: var(--text-mute); font-size: 12px; }
.warn-box { background: rgba(255,82,82,.06); border: 1px solid #ff525244; border-radius: 8px; padding: 10px 14px; font-size: 13px; color: #ff5252; }
</style>
