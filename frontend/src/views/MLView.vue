<script setup>
import { ref, reactive, computed, watch, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import api, { longTask, downloadFile } from '../api/client'
import { useToast } from '../stores/toast'
import { useResearchStore } from '../stores/research'
import EChart from '../components/EChart.vue'
import BoardSelect from '../components/BoardSelect.vue'

const { toast } = useToast()
const research = useResearchStore()
const router = useRouter()

const boards = ref(['all'])
const poolSize = ref(150)
const n = ref(3)
const hist = ref(500)
const modelType = ref('gbdt')
const nSplits = ref(5)
const gap = ref(5)
const useSnapshot = ref(false)
const assetClass = ref('a-share')   // a-share | future（期货主力连续合约池）
const trainStart = ref('')          // 训练集起始日（分时段训练，留空=最近 hist 天）
const trainEnd = ref('')            // 训练集结束日
const btStart = ref('')             // ML 回测验证区间起始日
const btEnd = ref('')               // ML 回测验证区间结束日
const btDirection = ref('long_short')  // long_short | long_only | short_only
const btBenchmark = ref('none')     // none | hs300 | zz500 | sse
const sectorOptions = ref([])

// 任务1：训练因子多选
const selectedFactors = ref([])
const factorPanelOpen = ref(false)

// 任务2：调参面板搜索/分组
const adjustSearch = ref('')
const adjGroupOpen = reactive({})

async function loadSectors() {
  try {
    const map = await api.get('/data/sectors')
    const names = [...new Set(Object.values(map || {}))].filter(Boolean).sort((a, b) => a.localeCompare(b, 'zh'))
    sectorOptions.value = names.map(n => ({ value: 'sector:' + n, label: n }))
  } catch (e) { /* 行业数据拉取失败不阻塞页面 */ }
}

const loading = ref(false)
const training = ref(false)
const trainMsg = ref('')
const result = ref(null)
const models = ref([])

const scoring = ref('')
const scoreResult = ref(null)
const btLoading = ref('')
const btResult = ref(null)

// U1 前端联动：设定训练/回测时间段后，按后端同款口径估算所需历史长度，用于提示与校验
const suggestedHist = computed(() => {
  const today = new Date(); today.setHours(0, 0, 0, 0)
  let need = 0
  for (const [s, e] of [[btStart.value, btEnd.value], [trainStart.value, trainEnd.value]]) {
    if (!s && !e) continue
    const from = s ? new Date(s + 'T00:00:00') : today
    const to = e ? new Date(e + 'T00:00:00') : today
    // 区间跨度 + 因子回看缓冲（60 缓冲 + 最长回看 240 ≈ 260）。
    // 两端都填=区间跨度；只填一端=该端到今天的距离；不再叠加 today→to 的重复距离。
    const spanDays = Math.max(0, Math.floor((to - from) / 86400000))
    need = Math.max(need, spanDays + 260)
  }
  return Math.max(0, Math.ceil(need))
})
const histInsufficient = computed(() => suggestedHist.value > Number(hist.value || 0))

// U1.3 自动回填：设定时间段后，若历史长度不足建议值，自动抬升；用户手动下调仍不足时禁用提交
watch(suggestedHist, v => {
  if (v > 0 && Number(hist.value || 0) < v) hist.value = v
})

// ---- 人工调参（特征权重 / 阈值） ----
const adjustPanel = ref('')
const adjustMeta = ref(null)
const featureWeights = reactive({})
const threshold = ref(null)
const lastAdjustId = ref('')
const adjustNote = ref('')
const cloneSaving = ref(false)
const monitorModelId = ref('')

async function loadMonitorActive() {
  try {
    const cfg = await api.get('/monitor/config')
    monitorModelId.value = cfg.mode === 'model' ? cfg.modelId : ''
  } catch (e) { /* 静默，盯盘模块可能未启动 */ }
}

async function openAdjust(m) {
  if (adjustPanel.value === m.id) { adjustPanel.value = ''; return }
  adjustPanel.value = m.id
  lastAdjustId.value = ''
  adjustNote.value = ''
  adjustSearch.value = ''
  try {
    adjustMeta.value = await api.get(`/ml/models/${m.id}/params`)
    for (const k of Object.keys(featureWeights)) delete featureWeights[k]
    for (const k of Object.keys(adjGroupOpen)) delete adjGroupOpen[k]
    threshold.value = adjustMeta.value.threshold ?? null
    const existingWeights = adjustMeta.value.featureWeights || {}
    const importanceMap = {}
    ;(adjustMeta.value.featureImportance || []).forEach(f => { importanceMap[f.feature] = f.importance })
    const featureNames = adjustMeta.value.featureNames || []
    for (const f of featureNames) {
      featureWeights[f] = existingWeights[f] ?? 1.0
    }
    // 默认展开有非零权重的分组
    const groupHasNonZero = {}
    for (const f of featureNames) {
      const mf = manualFeatures.value.find(m => m.key === f)
      const g = (mf ? mf.group : '其他') || '其他'
      if (!(g in groupHasNonZero)) groupHasNonZero[g] = false
      if ((featureWeights[f] ?? 0) !== 0) groupHasNonZero[g] = true
    }
    for (const [g, open] of Object.entries(groupHasNonZero)) {
      adjGroupOpen[g] = open
    }
  } catch (e) { toast(e.message); adjustPanel.value = '' }
}

function adjustPayload() {
  const fw = {}
  for (const [k, v] of Object.entries(featureWeights)) {
    const num = Number(v)
    if (k && v !== '' && v !== null && isFinite(num)) fw[k] = num
  }
  const th = (threshold.value === '' || threshold.value === null || threshold.value === undefined)
    ? null : Number(threshold.value)
  return { featureWeights: fw, threshold: th }
}

function featureLabel(key) {
  const m = manualFeatures.value.find(f => f.key === key)
  return m ? m.label : key
}

function selectAllFactors() { selectedFactors.value = manualFeatures.value.map(f => f.key) }
function deselectAllFactors() { selectedFactors.value = [] }

async function saveAdjust(m) {
  try {
    const res = await api.post(`/ml/models/${m.id}/adjust`, { ...adjustPayload(), saveArtifact: true })
    lastAdjustId.value = res.adjustId || ''
    adjustNote.value = res.effectNote || ''
    toast(lastAdjustId.value ? `调参配置已保存：${lastAdjustId.value}` : '调参已应用（未保存）')
  } catch (e) { toast(e.message) }
}

async function saveAsNewModel(m) {
  cloneSaving.value = true
  try {
    const res = await api.post(`/ml/models/${m.id}/adjust`, { ...adjustPayload(), saveAsNew: true })
    if (res.newModelId) {
      toast(`新模型已保存：${res.newModelId}`)
      await loadModels()
    } else if (res.cloneError) {
      toast(`另存失败：${res.cloneError}`)
    } else {
      toast('另存失败，请重试')
    }
  } catch (e) { toast(e.message) }
  finally { cloneSaving.value = false }
}

function evalConfig() {
  return {
    board: 'all', boards: boards.value, poolSize: Number(poolSize.value), n: Number(n.value),
    hist: Number(hist.value), modelType: modelType.value,
    nSplits: Number(nSplits.value), gap: Number(gap.value),
    useSnapshot: useSnapshot.value, assetClass: assetClass.value,
    startDate: trainStart.value || null, endDate: trainEnd.value || null,
    selectedFactors: selectedFactors.value.length ? selectedFactors.value : null,
  }
}

// jobs 轮询：页面隐藏暂停 + 间隔退避（1.5s→4s 上限）；离开页面停止，避免后台空转
let pollActive = true
async function pollJob(jobId, onProgress) {
  let delay = 1500
  while (pollActive) {
    if (document.hidden) { await new Promise(r => setTimeout(r, 1000)); continue }
    const j = await api.get(`/jobs/${jobId}`)
    onProgress && onProgress(j.progress, j.message)
    if (j.status === 'done') return j.result
    if (j.status === 'error') throw new Error(j.error || '任务失败')
    if (j.status === 'cancelled') throw new Error('任务已取消')
    await new Promise(r => setTimeout(r, delay))
    delay = Math.min(delay * 1.3, 4000)
  }
  throw new Error('任务已随页面离开而中止')
}

async function runEvaluate() {
  loading.value = true
  try {
    const { jobId } = await api.post('/jobs', { kind: 'ml-evaluate', config: evalConfig() })
    result.value = await pollJob(jobId)
    toast(`评估完成，OOS IC=${(result.value.oosIc || 0).toFixed(3)}`)
  } catch (e) { toast(e.message) }
  finally { loading.value = false }
}

async function runTrain() {
  training.value = true
  trainMsg.value = '提交训练任务…'
  try {
    const { jobId } = await api.post('/jobs', { kind: 'ml-train', config: evalConfig() })
    const res = await pollJob(jobId, (p, m) => trainMsg.value = m || `进度 ${p || 0}%`)
    result.value = res.evaluation
    await loadModels()
    // 写入跨页共享 store，主回测页 onMounted 消费并自动选中该模型（打通 ML→主回测闭环）
    research.setCurrentModel(res.model)
    toast(`训练完成，模型 ${res.model.id} 已落盘`)
  } catch (e) { toast(e.message) }
  finally { training.value = false; trainMsg.value = '' }
}

async function loadModels() {
  try { models.value = await api.get('/ml/models') }
  catch (e) { toast(e.message) }
}

async function deleteModel(id) {
  if (!confirm('删除该模型文件？')) return
  try { await api.delete(`/ml/models/${id}`); models.value = models.value.filter(m => m.id !== id) }
  catch (e) { toast(e.message) }
}

// 把模型带到主回测页：写入共享 store 后跳转，BacktestView onMounted 自动选中
function gotoBacktest(m) {
  research.setCurrentModel(m)
  router.push('/backtest')
}

// 用模型对候选池最新截面打分（ML→选股闭环，支持人工调参权重/阈值）
const scoreLongList = ref([])     // 做多候选（最高分）
const scoreShortList = ref([])    // 做空候选（最低分，allowShort 时）
const scoreDirection = ref('long_short')
async function runScore(m) {
  scoring.value = m.id
  scoreResult.value = null
  scoreLongList.value = []
  scoreShortList.value = []
  try {
    const payload = { modelId: m.id, board: 'all', boards: boards.value, poolSize: Number(poolSize.value), assetClass: assetClass.value }
    if (adjustPanel.value === m.id) {
      const adj = adjustPayload()
      if (Object.keys(adj.featureWeights).length || adj.threshold !== null) payload.adjust = adj
    }
    const res = await longTask('/ml/score', payload)
    scoreResult.value = res?.scores ?? res
    scoreLongList.value = res?.longList || []
    scoreShortList.value = res?.shortList || []
    scoreDirection.value = res?.direction || 'long_short'
    toast(`打分完成，共 ${scoreResult.value.length} 只`)
  } catch (e) { toast(e.message) }
  finally { scoring.value = '' }
}

// 用模型预测分做分层回测（ML→回测闭环，走 jobs 异步；btStart/btEnd 限定验证区间）
async function runMLBacktest(m) {
  btLoading.value = m.id
  btResult.value = null
  // 模型声明方向作为默认值（UI 可在回测卡片下拉中覆盖）
  if (m.direction) btDirection.value = m.direction
  try {
    const { jobId } = await api.post('/jobs', { kind: 'ml-backtest', config: {
      modelId: m.id, board: 'all', boards: boards.value, poolSize: Number(poolSize.value),
      n: Number(n.value), hist: Number(hist.value), assetClass: assetClass.value,
      startDate: btStart.value || null, endDate: btEnd.value || null,
      direction: btDirection.value, benchmark: btBenchmark.value,
    }})
    btResult.value = await pollJob(jobId)
    if (btResult.value?.ok === false) {
      toast(btResult.value.hint || btResult.value.error || '回测区间无有效调仓日')
      btResult.value = null
      return
    }
    if (btResult.value?.snapshotStartNote) toast(btResult.value.snapshotStartNote)
    if (btResult.value?.inSampleWarning) toast(btResult.value.inSampleWarning)
    toast(`ML 回测完成，调仓 ${btResult.value.rebalanceCount} 次`)
  } catch (e) { toast(e.message) }
  finally { btLoading.value = '' }
}

// P4：导入外部模型文件（joblib，含 model + feature_names）
const importInput = ref(null)
const importing = ref(false)
async function importModel() {
  const el = importInput.value
  if (!el || !el.files || !el.files.length) { toast('请先选择模型文件（.joblib/.pkl）'); return }
  importing.value = true
  try {
    const fd = new FormData()
    fd.append('file', el.files[0])
    const meta = await api.post('/ml/models/import', fd, { timeout: 120000 })
    toast(`已导入模型 ${meta.id}`)
    await loadModels()
    el.value = ''
  } catch (e) { toast(e.message) }
  finally { importing.value = false }
}

// P4：人造/手动模型（手工指定因子权重，不依赖自动训练）
const manualFeatures = ref([])
const manualName = ref('')
const manualThreshold = ref(0)
const manualRule = ref('')
const manualBullRule = ref('')
const manualBearRule = ref('')
const manualWeights = reactive({})
const manualDirection = ref('long_short')
const manualAllowShort = ref(true)
const creatingManual = ref(false)
async function loadManualFeatures() {
  try {
    const data = await api.get('/ml/manual/features')
    manualFeatures.value = data
    for (const f of data) if (!(f.key in manualWeights)) manualWeights[f.key] = 0
  } catch (e) { /* 静默 */ }
}
async function saveManualModel() {
  if (!manualName.value.trim()) { toast('请输入模型名称'); return }
  const fw = {}
  for (const [k, v] of Object.entries(manualWeights)) {
    const num = Number(v)
    if (num !== 0 && isFinite(num)) fw[k] = num
  }
  if (!Object.keys(fw).length) { toast('请至少为一个因子设置非零权重'); return }
  creatingManual.value = true
  try {
    const meta = await api.post('/ml/models/manual', {
      name: manualName.value.trim(),
      featureWeights: fw,
      threshold: manualThreshold.value === '' || manualThreshold.value === null ? 0 : Number(manualThreshold.value),
      rule: manualRule.value.trim(),
      bullRule: manualBullRule.value.trim(),
      bearRule: manualBearRule.value.trim(),
      direction: manualDirection.value,
      allowShort: manualAllowShort.value,
    })
    toast(`人造模型已创建：${meta.id}`)
    await loadModels()
    manualName.value = ''
  } catch (e) { toast(e.message) }
  finally { creatingManual.value = false }
}

// 外部模型导入引导：查看特征模板
const templateVisible = ref(false)
const importTemplate = ref(null)
async function showImportTemplate() {
  try {
    importTemplate.value = await api.get('/ml/models/import-template')
    templateVisible.value = true
  } catch (e) { toast(e.message) }
}

// ML 回测结果导出（复用主回测页的 /reports/backtest，旧版 ML 回测无导出能力）
async function exportMLBacktest(fmt) {
  if (!btResult.value) return
  try {
    const payload = {
      format: fmt, factorLabel: btResult.value.factorLabel,
      config: btResult.value.config || {}, metrics: btResult.value.metrics,
      benchmark: btResult.value.benchmark, groupSummary: btResult.value.groupSummary,
      longShort: btResult.value.longShort, icSeries: btResult.value.icSeries,
      positionLedger: btResult.value.positionLedger || [],
      featureImportance: btResult.value.featureImportance || [],
      actualHistDays: btResult.value.actualHistDays,
      effectiveStart: btResult.value.effectiveStart,
      effectiveEnd: btResult.value.effectiveEnd,
      signalMode: btResult.value.signalMode || 'group',
      bullRule: btResult.value.bullRule || '',
      bearRule: btResult.value.bearRule || '',
      topAttribution: btResult.value.topAttribution || null,
      direction: btResult.value.direction || 'long_short',
      survivorshipBiasWarning: btResult.value.survivorshipBiasWarning || '',
      snapshotWarning: btResult.value.snapshotWarning || '',
      snapshotStartNote: btResult.value.snapshotStartNote || '',
      inSampleWarning: btResult.value.inSampleWarning || '',
      histWarning: btResult.value.histWarning || '',
      icIr: btResult.value.icIr ?? null,
      icTStat: btResult.value.icTStat ?? null,
      icPValue: btResult.value.icPValue ?? null,
      yearlyReturns: btResult.value.yearlyReturns || [],
      stockContribution: btResult.value.stockContribution || [],
      bucketDates: btResult.value.bucketDates || [],
    }
    const ext = fmt === 'html' ? 'html' : fmt === 'pdf' ? 'pdf' : 'xlsx'
    await downloadFile('/reports/backtest', payload, `ml_backtest.${ext}`)
  } catch (e) { toast(e.message) }
}

const fmt = v => v == null ? '-' : Number(v).toFixed(3)
const fmtPct = v => v == null ? '-' : (Number(v) * 100).toFixed(2) + '%'

const impOption = computed(() => {
  if (!result.value?.featureImportance) return {}
  const data = result.value.featureImportance.slice(0, 15).map(f => ({ name: f.feature, value: f.importance }))
  return {
    grid: { left: 100, right: 20, top: 20, bottom: 20 },
    tooltip: {},
    xAxis: { type: 'value' },
    yAxis: { type: 'category', data: data.map(d => d.name) },
    series: [{ type: 'bar', data: data.map(d => d.value), itemStyle: { color: '#4f8cff' } }]
  }
})

// 任务1：因子按 group 分组
const groupedFeatures = computed(() => {
  const groups = {}
  for (const f of manualFeatures.value) {
    const g = f.group || '其他'
    if (!groups[g]) groups[g] = []
    groups[g].push(f)
  }
  return groups
})

// 任务2：调参因子增强（含 importance、group、label）
const adjustFeaturesEnriched = computed(() => {
  if (!adjustMeta.value) return []
  const impMap = {}
  const fi = adjustMeta.value.featureImportance || adjustMeta.value.featureImportances || []
  for (const item of fi) {
    if (item.feature !== undefined) impMap[item.feature] = item.importance ?? 0
  }
  return (adjustMeta.value.featureNames || []).map(f => ({
    key: f,
    label: featureLabel(f),
    group: (manualFeatures.value.find(mf => mf.key === f) || {}).group || '其他',
    importance: impMap[f] ?? 0,
  })).sort((a, b) => b.importance - a.importance)
})

const filteredAdjustFeatures = computed(() => {
  const q = (adjustSearch.value || '').trim().toLowerCase()
  if (!q) return adjustFeaturesEnriched.value
  return adjustFeaturesEnriched.value.filter(f =>
    f.label.toLowerCase().includes(q) || f.key.toLowerCase().includes(q)
  )
})

const groupedAdjustFeatures = computed(() => {
  const groups = []
  const map = {}
  for (const f of filteredAdjustFeatures.value) {
    const g = f.group
    if (!map[g]) { map[g] = []; groups.push(g) }
    map[g].push(f)
  }
  return groups.map(g => ({ group: g, features: map[g] }))
})

function toggleAdjGroup(group) {
  adjGroupOpen[group] = !adjGroupOpen[group]
}

const lsOption = computed(() => {
  if (!btResult.value?.longShort?.length) return {}
  const pts = btResult.value.longShort
  const dir = btResult.value.direction || 'long_short'
  const series = [{
    name: dir === 'long_only' ? '多头累计' : dir === 'short_only' ? '空头累计' : '多空累计',
    type: 'line', smooth: true,
    data: pts.map(p => Number((p.cum * 100).toFixed(3))),
    itemStyle: { color: '#4f8cff' }, areaStyle: { color: 'rgba(79,140,255,.12)' },
  }]
  if (dir === 'long_short') {
    series.push({
      name: '多头腿', type: 'line', smooth: true,
      data: pts.map(p => Number(((p.topCum ?? p.cum) * 100).toFixed(3))),
      itemStyle: { color: '#21c08b' },
    })
    series.push({
      name: '空头腿', type: 'line', smooth: true,
      data: pts.map(p => Number(((p.bottomCum ?? 0) * 100).toFixed(3))),
      itemStyle: { color: '#6c5ce7' },
    })
  }
  return {
    grid: { left: 50, right: 20, top: 20, bottom: 30 },
    tooltip: { trigger: 'axis' },
    legend: { top: 0 },
    xAxis: { type: 'category', data: pts.map(p => p.date) },
    yAxis: { type: 'value', axisLabel: { formatter: v => (v * 100).toFixed(1) + '%' } },
    series,
  }
})

// U3/U4 持仓与换手变化图（positionLedger 后端直算 longCount/shortCount/turnover）
const posOption = computed(() => {
  const pl = btResult.value?.positionLedger || []
  if (!pl.length) return {}
  return {
    grid: { left: 50, right: 50, top: 30, bottom: 30 },
    tooltip: { trigger: 'axis' },
    legend: { top: 0 },
    xAxis: { type: 'category', data: pl.map(p => p.date) },
    yAxis: [
      { type: 'value', name: '持仓数' },
      { type: 'value', name: '换手' },
    ],
    series: [
      { name: '多头持仓数', type: 'line', data: pl.map(p => p.longCount ?? 0), itemStyle: { color: '#21c08b' } },
      { name: '空头持仓数', type: 'line', data: pl.map(p => p.shortCount ?? 0), itemStyle: { color: '#6c5ce7' } },
      { name: '换手股数', type: 'bar', yAxisIndex: 1, data: pl.map(p => p.turnover ?? 0), itemStyle: { color: 'rgba(79,140,255,.4)' } },
    ],
  }
})

onMounted(() => { loadModels(); loadSectors(); loadManualFeatures(); loadMonitorActive() })
onUnmounted(() => { pollActive = false })
</script>

<template>
  <div>
    <div class="card">
      <div class="card-head"><h3>机器学习 · 因子收益预测</h3><span class="hint">GBDT + 时序 Walk-Forward CV，防前视偏差</span></div>
      <div class="panel-toolbar">
        <div class="field"><label>资产类别</label>
          <select v-model="assetClass">
            <option value="a-share">A股</option>
            <option value="future">期货（主力连续）</option>
          </select>
        </div>
        <div class="field"><label>板块</label>
          <BoardSelect v-model="boards" :sectors="sectorOptions" />
        </div>
        <div class="field"><label>候选池</label><input v-model="poolSize" type="number" /></div>
        <div class="field"><label>预测期N(日)</label><input v-model="n" type="number" /></div>
        <div class="field"><label>历史长度</label><input v-model="hist" type="number" /></div>
        <div class="field"><label>CV折数</label><input v-model="nSplits" type="number" /></div>
        <div class="field"><label>Gap</label><input v-model="gap" type="number" /></div>
      </div>
      <div class="panel-toolbar" style="margin-top:10px">
        <div class="field"><label>训练起始日</label><input v-model="trainStart" type="date" /></div>
        <div class="field"><label>训练结束日</label><input v-model="trainEnd" type="date" /></div>
        <span class="hint">留空=用最近 hist 天全部样本；两端都填=只训练该区间，只填起始日=训练至今天，只填结束日=取最近 hist 天截至该日。限定时间段后请确保「历史长度」≥ 时间段跨度+260日（不足时已自动回填）<template v-if="suggestedHist > 0">｜建议历史长度 ≥ {{ suggestedHist }} 日<strong v-if="histInsufficient" style="color:#e6a817">（当前 {{ hist }} 不足，请增大历史长度）</strong></template></span>
      </div>
      <!-- 任务1：因子选择折叠面板 -->
      <div class="factor-select-box">
        <button class="btn-ghost sm" @click="factorPanelOpen = !factorPanelOpen">
          因子选择 ({{ selectedFactors.length }}/{{ manualFeatures.length }})
          <span class="arrow" :class="{ open: factorPanelOpen }">▾</span>
        </button>
        <div v-if="factorPanelOpen" class="factor-panel">
          <div class="factor-actions">
            <button class="btn-ghost sm" @click="selectAllFactors">全选</button>
            <button class="btn-ghost sm" @click="deselectAllFactors">取消全选</button>
          </div>
          <div class="factor-groups">
            <div v-for="(features, group) in groupedFeatures" :key="group" class="factor-group">
              <div class="factor-group-title">{{ group }} ({{ features.length }})</div>
              <div class="factor-grid">
                <label v-for="f in features" :key="f.key" class="factor-checkbox">
                  <input type="checkbox" :value="f.key" v-model="selectedFactors" />
                  <span>{{ f.label }}</span>
                </label>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="panel-toolbar" style="margin-top:10px">
        <button class="btn-ghost" :disabled="loading || histInsufficient" :title="histInsufficient ? '历史长度不足，请增大历史长度或缩短时间段' : ''" @click="runEvaluate">{{ loading ? '评估中…' : '评估(CV)' }}</button>
        <button class="btn-primary" :disabled="training || histInsufficient" :title="histInsufficient ? '历史长度不足，请增大历史长度或缩短时间段' : ''" @click="runTrain">{{ training ? '训练中…' : '训练并落盘' }}</button>
        <label style="margin-left:12px;font-size:13px;color:var(--text-mute)"><input type="checkbox" v-model="useSnapshot" /> 含全部快照因子（PE/PB/ROE/北向等，前视，探索用）</label>
        <span v-if="trainMsg" class="hint" style="margin-left:8px">{{ trainMsg }}</span>
      </div>
    </div>

    <div v-if="result" class="card">
      <div class="card-head"><h3>OOS 评估</h3><span class="hint">样本量 {{ result.nSamples }} · 特征 {{ result.nFeatures }}</span></div>
      <div class="kpi-row">
        <div class="kpi"><div class="n">{{ fmt(result.oosIc) }}</div><div class="l">OOS IC</div></div>
        <div class="kpi"><div class="n">{{ fmt(result.oosRankIc) }}</div><div class="l">OOS RankIC</div></div>
        <div class="kpi"><div class="n">{{ fmtPct(result.oosLongShort) }}</div><div class="l">多空收益</div></div>
        <div class="kpi"><div class="n">{{ fmt(result.oosSharpe) }}</div><div class="l">OOS Sharpe</div></div>
      </div>
      <table class="data-table">
        <thead><tr><th>折</th><th>训练量</th><th>测试量</th><th>IC</th><th>RankIC</th><th>多空</th><th>RMSE</th></tr></thead>
        <tbody>
          <tr v-for="f in result.folds" :key="f.fold">
            <td>{{ f.fold }}</td><td>{{ f.trainSize }}</td><td>{{ f.testSize }}</td>
            <td>{{ fmt(f.ic) }}</td><td>{{ fmt(f.rankIc) }}</td>
            <td class="up">{{ fmtPct(f.longShort) }}</td><td>{{ fmt(f.rmse) }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div v-if="result?.featureImportance?.length" class="card">
      <h3>特征重要性 Top15</h3>
      <EChart :option="impOption" style="height:380px" />
    </div>

    <div class="card">
      <div class="card-head"><h3>新建人造模型（手搓模型）</h3><span class="hint">手动为每个因子设定权重合成预测分，保存后与训练模型同等可用（打分/回测/盯盘）</span></div>
      <div class="panel-toolbar" style="margin-top:10px">
        <div class="field"><label>模型名称</label><input v-model="manualName" placeholder="如：动量+低波 组合" /></div>
        <div class="field"><label>阈值偏移</label><input v-model="manualThreshold" type="number" step="0.01" placeholder="0" /></div>
        <div class="field">
          <label>交易方向</label>
          <select v-model="manualDirection">
            <option value="long_short">多空对冲</option>
            <option value="long_only">仅做多</option>
            <option value="short_only">仅做空</option>
          </select>
        </div>
        <div class="field">
          <label>允许做空</label>
          <input v-model="manualAllowShort" type="checkbox" />
        </div>
        <div class="field grow"><label>规则说明（可选）</label><input v-model="manualRule" placeholder="如：动量>0.1 且 RSI<30 买入" /></div>
        <button class="btn-primary" :disabled="creatingManual" @click="saveManualModel">{{ creatingManual ? '创建中…' : '创建人造模型' }}</button>
      </div>
      <div class="panel-toolbar" style="margin-top:10px">
        <div class="field grow"><label>看多规则（可选，回测离散买卖）</label><input v-model="manualBullRule" placeholder="如：scorePct>=0.8 and rsi<70" /></div>
        <div class="field grow"><label>看空规则（可选，回测离散买卖）</label><input v-model="manualBearRule" placeholder="如：scorePct<=0.2 or momentum<-0.05" /></div>
      </div>
      <p class="hint" style="margin-top:8px">规则支持变量：scorePct（预测分截面分位 0~1）+ 因子名（如 rsi/momentum/volatility），运算 + - * / &gt; &lt; &gt;= &lt;= == and or not 与括号。留空则回测退回分位分组。</p>
      <div class="manual-features">
        <div class="mf-row" v-for="f in manualFeatures" :key="f.key">
          <span class="mf-name" :title="f.key">{{ f.label }}</span>
          <span class="mf-key">{{ f.key }}</span>
          <input class="mf-weight" v-model="manualWeights[f.key]" type="number" step="0.1" placeholder="权重(0=不参与)" />
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-head"><h3>已训练模型</h3><span class="hint">joblib 落盘，可用 joblib.load 复用</span></div>
      <div class="panel-toolbar" style="margin-top:10px">
        <input ref="importInput" type="file" accept=".joblib,.pkl,.pickle" class="file-input" />
        <button class="btn-ghost sm" :disabled="importing" @click="importModel">{{ importing ? '导入中…' : '导入模型文件' }}</button>
        <button class="btn-ghost sm" @click="showImportTemplate">查看特征模板/示例</button>
        <span class="hint">支持本平台训练产物或同构 joblib（须含 model + feature_names，缺失预处理参数时按恒等变换兜底）</span>
      </div>
      <div v-if="templateVisible && importTemplate" class="tpl-box">
        <div class="card-head"><h4>外部模型导入模板</h4>
          <button class="btn-ghost sm" @click="templateVisible = false">收起</button>
        </div>
        <p class="hint">{{ importTemplate.note }}</p>
        <p class="hint">平台特征共 {{ importTemplate.featureNames.length }} 个，须与模型 feature_names 完全一致。</p>
        <pre class="tpl-code">{{ importTemplate.sampleCode }}</pre>
      </div>
      <div class="panel-toolbar" style="margin-top:10px">
        <div class="field"><label>回测起始日</label><input v-model="btStart" type="date" /></div>
        <div class="field"><label>回测结束日</label><input v-model="btEnd" type="date" /></div>
        <div class="field">
          <label>交易方向</label>
          <select v-model="btDirection">
            <option value="long_short">多空对冲</option>
            <option value="long_only">仅做多</option>
            <option value="short_only">仅做空</option>
          </select>
        </div>
        <div class="field">
          <label>基准</label>
          <select v-model="btBenchmark">
            <option value="none">无</option>
            <option value="hs300">沪深300</option>
            <option value="zz500">中证500</option>
            <option value="sse">上证指数</option>
          </select>
        </div>
        <span class="hint">「ML回测」的验证区间（分时段验证），留空=整个历史；只填起始日=回测至今天，只填结束日=取最近 hist 天截至该日<template v-if="suggestedHist > 0">｜建议历史长度 ≥ {{ suggestedHist }} 日<strong v-if="histInsufficient" style="color:#e6a817">（当前 {{ hist }} 不足，请增大历史长度）</strong></template></span>
      </div>
      <div v-if="!models.length" class="empty-hint">暂无模型，点击上方「训练并落盘」「创建人造模型」或导入模型文件</div>
      <table v-else class="data-table">
        <thead><tr><th>模型ID</th><th>类型</th><th>因子数</th><th>可回测</th><th>操作</th></tr></thead>
        <tbody>
          <tr v-for="m in models" :key="m.id" :class="{ 'row-disabled': m.computable === false }">
            <td>
              {{ m.id }}
              <span v-if="m.computable === false" class="badge-warn" :title="'未知特征: ' + (m.unknownFeatures||[]).join(', ')">因子不可计算</span>
              <span v-if="monitorModelId === m.id" class="badge-monitor" title="此模型正在盯盘调度中使用">🔍 盯盘中</span>
            </td>
            <td class="muted">{{ m.modelType || '?' }}</td>
            <td class="muted">{{ m.nFeatures ?? '?' }}</td>
            <td><span :class="m.computable === false ? 'tag-fail' : 'tag-ok'">{{ m.computable === false ? '否' : '是' }}</span></td>
            <td>
              <button class="btn-ghost sm" :disabled="scoring===m.id || !m.computable" :title="!m.computable ? '模型因子不可从K线计算，无法回测' : ''" @click="runScore(m)">{{ scoring===m.id ? '打分中…' : '打分选股' }}</button>
              <button class="btn-ghost sm" :disabled="btLoading===m.id || !m.computable || histInsufficient" :title="histInsufficient ? '历史长度不足，请增大历史长度或缩短时间段' : (!m.computable ? '模型因子不可从K线计算，无法回测' : '')" @click="runMLBacktest(m)">{{ btLoading===m.id ? '回测中…' : 'ML回测' }}</button>
              <button class="btn-ghost sm" @click="openAdjust(m)">{{ adjustPanel===m.id ? '收起调参' : '调参' }}</button>
              <button class="btn-ghost sm" :disabled="!m.computable" :title="!m.computable ? '模型因子不可从K线计算，无法回测' : ''" @click="gotoBacktest(m)">去主回测</button>
              <button class="btn-ghost sm danger" @click="deleteModel(m.id)">删除</button>
            </td>
          </tr>
        </tbody>
      </table>
      <div v-if="adjustPanel && adjustMeta" class="adjust-box">
        <div class="card-head">
          <h4>人工调参：{{ adjustPanel }}</h4>
          <span class="hint">调整特征权重或预测阈值，保存后打分/回测即时生效</span>
        </div>
        <div class="adj-threshold">
          <label>预测阈值偏移</label>
          <input v-model="threshold" type="number" step="0.01" placeholder="默认 0" />
        </div>
        <!-- 任务2：搜索框 -->
        <div class="adj-search">
          <input v-model="adjustSearch" type="text" placeholder="搜索因子名称或 key…" class="adj-search-input" />
        </div>
        <!-- 任务2：按分组折叠 + importance 排序 -->
        <div class="adj-groups">
          <div v-for="g in groupedAdjustFeatures" :key="g.group" class="adj-group">
            <div class="adj-group-title" @click="toggleAdjGroup(g.group)">
              <span class="adj-group-arrow" :class="{ open: adjGroupOpen[g.group] }">▸</span>
              <span>{{ g.group }}</span>
              <span class="adj-group-count">{{ g.features.length }} 因子</span>
            </div>
            <div v-if="adjGroupOpen[g.group]" class="adj-group-features">
              <div class="adj-row" v-for="f in g.features" :key="f.key">
                <span class="adj-name" :title="f.key">{{ f.label }}</span>
                <span class="adj-imp" v-if="f.importance" :title="'importance: ' + f.importance.toFixed(4)">
                  {{ f.importance.toFixed(3) }}
                </span>
                <input class="adj-weight" v-model="featureWeights[f.key]" type="number" step="0.1" min="0" />
              </div>
            </div>
          </div>
          <div v-if="!filteredAdjustFeatures.length" class="adj-empty">无匹配因子</div>
        </div>
        <div class="panel-toolbar">
          <button class="btn-primary sm" @click="saveAdjust({ id: adjustPanel })">保存调参配置</button>
          <button class="btn-ghost sm" :disabled="cloneSaving" @click="saveAsNewModel({ id: adjustPanel })">{{ cloneSaving ? '另存中…' : '另存为新模型' }}</button>
          <span v-if="lastAdjustId" class="hint">已保存 adjustId：{{ lastAdjustId }}（打分/回测自动生效）</span>
        </div>
        <div v-if="adjustNote" class="adj-note">{{ adjustNote }}</div>
      </div>
    </div>

    <div v-if="scoreResult" class="card">
      <div class="card-head"><h3>模型截面打分</h3><span class="hint">
        最新截面预测分排序（Top 30）· 方向 {{ scoreDirection }}
      </span></div>
      <div class="score-grid">
        <div class="score-col">
          <h4 class="score-title long">做多候选（最高分）</h4>
          <table class="data-table">
            <thead><tr><th>排名</th><th>代码</th><th>名称</th><th>预测分</th></tr></thead>
            <tbody>
              <tr v-for="(r,i) in scoreLongList.slice(0,30)" :key="r.code">
                <td>{{ i + 1 }}</td><td>{{ r.code }}</td><td>{{ r.name }}</td>
                <td class="up">{{ fmt(r.score) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div class="score-col" v-if="scoreShortList.length">
          <h4 class="score-title short">做空候选（最低分）</h4>
          <table class="data-table">
            <thead><tr><th>排名</th><th>代码</th><th>名称</th><th>预测分</th></tr></thead>
            <tbody>
              <tr v-for="(r,i) in scoreShortList.slice(0,30)" :key="r.code">
                <td>{{ i + 1 }}</td><td>{{ r.code }}</td><td>{{ r.name }}</td>
                <td class="down">{{ fmt(r.score) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <div v-if="btResult" class="card">
      <div class="card-head"><h3>ML 信号分层回测</h3><span class="hint">
        调仓 {{ btResult.rebalanceCount }} 次 · 有效股票 {{ btResult.effectiveStocks }}
        <button class="btn-ghost sm" style="margin-left:12px" @click="exportMLBacktest('html')">导出HTML</button>
        <button class="btn-ghost sm" style="margin-left:6px" @click="exportMLBacktest('excel')">导出Excel</button>
        <button class="btn-ghost sm" style="margin-left:6px" @click="exportMLBacktest('pdf')">导出PDF</button>
      </span></div>
      <div class="kpi-row">
        <div class="kpi"><div class="n">{{ fmtPct(btResult.metrics.cumulativeReturn) }}</div><div class="l">累计收益</div></div>
        <div class="kpi"><div class="n">{{ fmtPct(btResult.metrics.annualizedReturn) }}</div><div class="l">年化</div></div>
        <div class="kpi"><div class="n">{{ fmt(btResult.metrics.sharpe) }}</div><div class="l">Sharpe</div></div>
        <div class="kpi"><div class="n">{{ fmtPct(btResult.metrics.maxDrawdown) }}</div><div class="l">最大回撤</div></div>
        <div class="kpi"><div class="n">{{ fmt(btResult.meanIc) }}</div><div class="l">IC</div></div>
      </div>
      <EChart :option="lsOption" style="height:340px" />
    </div>

    <div v-if="btResult?.positionLedger?.length" class="card">
      <div class="card-head"><h3>持仓与调仓变动</h3><span class="hint">每期多头/空头持仓数量及换手股数；明细为最近 10 期相对上期的进出标的</span></div>
      <EChart :option="posOption" style="height:300px" />
      <table class="data-table" style="margin-top:12px">
        <thead><tr><th>调仓日</th><th>多头持仓</th><th>空头持仓</th><th>换手</th><th>新进多头</th><th>退出多头</th><th>新进空头</th><th>退出空头</th></tr></thead>
        <tbody>
          <tr v-for="p in btResult.positionLedger.slice(-10)" :key="p.date">
            <td>{{ p.date }}</td>
            <td>{{ p.longCount ?? '-' }}</td>
            <td>{{ p.shortCount ?? '-' }}</td>
            <td>{{ p.turnover ?? '-' }}</td>
            <td>{{ (p.longAdded || []).join(', ') || '-' }}</td>
            <td>{{ (p.longRemoved || []).join(', ') || '-' }}</td>
            <td>{{ (p.shortAdded || []).join(', ') || '-' }}</td>
            <td>{{ (p.shortRemoved || []).join(', ') || '-' }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.data-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
.data-table th, .data-table td { border: 1px solid var(--border); padding: 9px 11px; text-align: left; }
.data-table th { background: var(--card-2); color: var(--text-dim); font-weight: 600; }
.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 14px 0; }
.kpi { background: var(--card-2); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }
.kpi .n { font-size: 20px; font-weight: 700; color: var(--accent); }
.kpi .l { color: var(--text-mute); font-size: 12px; margin-top: 4px; }
.hint { color: var(--text-mute); font-size: 12px; font-weight: 400; }
.card-head { display: flex; justify-content: space-between; align-items: baseline; }
.muted { color: var(--text-mute); font-size: 12px; }
.btn-ghost.sm.danger { color: #ff6b6b; }
.adj-note { margin-top: 10px; padding: 8px 12px; background: rgba(79,140,255,.08); border: 1px solid rgba(79,140,255,.2); border-radius: 8px; font-size: 12px; color: var(--text-dim); }
.btn-ghost.sm { padding: 4px 10px; font-size: 12px; }
.score-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 8px; }
.score-grid > .score-col:only-child { grid-column: 1 / -1; }
.score-title { margin: 0 0 4px; font-size: 13px; font-weight: 600; }
.score-title.long { color: #00c853; }
.score-title.short { color: #ff5252; }
.up { color: #ff5252; }
.down { color: #00c853; }
@media (max-width: 720px) { .score-grid { grid-template-columns: 1fr; } }
.adjust-box { border-top: 1px solid var(--border); padding: 16px 22px; }
.adj-threshold { display: flex; align-items: center; gap: 10px; margin: 10px 0; }
.adj-threshold input { width: 120px; }
/* 旧 .adj-features 替换为 adj-groups */
.adj-search { margin: 10px 0; }
.adj-search-input { width: 100%; max-width: 360px; padding: 6px 10px; font-size: 13px; border: 1px solid var(--border); border-radius: 6px; background: var(--bg-2); color: var(--text); }
.adj-groups { max-height: 420px; overflow: auto; }
.adj-group { margin-bottom: 4px; }
.adj-group-title { display: flex; align-items: center; gap: 6px; padding: 8px 12px; background: var(--card-2); border: 1px solid var(--border); border-radius: 8px; cursor: pointer; user-select: none; font-size: 13px; font-weight: 600; color: var(--text-dim); }
.adj-group-title:hover { background: var(--bg-2); }
.adj-group-arrow { display: inline-block; width: 14px; font-size: 11px; transition: transform .15s; color: var(--text-mute); }
.adj-group-arrow.open { transform: rotate(90deg); }
.adj-group-count { font-size: 11px; color: var(--text-mute); font-weight: 400; margin-left: auto; }
.adj-group-features { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 6px; padding: 8px 0 8px 20px; }
.adj-row { display: flex; align-items: center; gap: 8px; background: var(--bg-2); border: 1px solid var(--border); border-radius: 8px; padding: 6px 10px; }
.adj-name { flex: 1; font-size: 12px; color: var(--text-dim); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.adj-imp { font-size: 10px; color: var(--accent); font-family: monospace; min-width: 40px; text-align: right; }
.adj-weight { width: 64px; }
.adj-empty { text-align: center; padding: 20px; color: var(--text-mute); font-size: 13px; }
/* 因子选择面板 */
.factor-select-box { margin-top: 10px; }
.factor-select-box .arrow { display: inline-block; margin-left: 4px; font-size: 10px; transition: transform .15s; }
.factor-select-box .arrow.open { transform: rotate(180deg); }
.factor-panel { border: 1px solid var(--border); border-radius: 10px; padding: 12px 16px; margin-top: 8px; background: var(--bg-2); }
.factor-actions { display: flex; gap: 8px; margin-bottom: 10px; }
.factor-groups { max-height: 360px; overflow: auto; }
.factor-group { margin-bottom: 10px; }
.factor-group-title { font-size: 13px; font-weight: 600; color: var(--text-dim); margin-bottom: 6px; padding-bottom: 4px; border-bottom: 1px solid var(--border); }
.factor-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 4px; }
/* 模型可回测状态标记 */
.row-disabled td { opacity: 0.55; }
.badge-warn { display: inline-block; margin-left: 6px; padding: 1px 6px; border-radius: 4px; background: rgba(255,107,107,.15); color: #ff6b6b; font-size: 10px; font-weight: 600; }
.badge-monitor { display: inline-block; margin-left: 6px; padding: 1px 6px; border-radius: 4px; background: rgba(79,140,255,.15); color: #4f8cff; font-size: 10px; font-weight: 600; }
.tag-ok { color: #22c55e; font-weight: 600; font-size: 12px; }
.tag-fail { color: #ff6b6b; font-weight: 600; font-size: 12px; }
.factor-checkbox { display: flex; align-items: center; gap: 5px; font-size: 12px; color: var(--text-dim); cursor: pointer; white-space: nowrap; }
.factor-checkbox input { margin: 0; }
.panel-toolbar { display: flex; align-items: center; gap: 14px; margin-top: 12px; }
.file-input { font-size: 12px; color: var(--text-dim); max-width: 320px; }
.manual-features { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 8px; margin-top: 12px; max-height: 300px; overflow: auto; }
.mf-row { display: flex; align-items: center; gap: 8px; background: var(--bg-2); border: 1px solid var(--border); border-radius: 8px; padding: 6px 10px; }
.mf-name { flex: 1; font-size: 12px; color: var(--text-dim); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.mf-key { font-size: 11px; color: var(--text-mute); font-family: monospace; }
.mf-weight { width: 84px; }
.tpl-box { border-top: 1px solid var(--border); padding: 14px 20px; }
.tpl-code { background: var(--bg-2); border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-size: 12px; overflow-x: auto; color: var(--text-dim); }
</style>
