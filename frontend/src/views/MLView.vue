<script setup>
import { ref, reactive, computed, onMounted, onUnmounted } from 'vue'
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

// ---- 人工调参（特征权重 / 阈值） ----
const adjustPanel = ref('')
const adjustMeta = ref(null)
const featureWeights = reactive({})
const threshold = ref(null)
const lastAdjustId = ref('')
const adjustNote = ref('')

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
    const featureNames = adjustMeta.value.featureNames || []
    for (const f of featureNames) {
      featureWeights[f] = existingWeights[f] ?? 1
    }
    // 默认展开有非零权重的分组
    const groupHasNonZero = {}
    for (const f of featureNames) {
      const mf = manualFeatures.value.find(m => m.key === f)
      const g = (mf ? mf.group : '其他') || '其他'
      if (!(g in groupHasNonZero)) groupHasNonZero[g] = false
      if ((featureWeights[f] ?? 1) !== 1) groupHasNonZero[g] = true
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
async function runScore(m) {
  scoring.value = m.id
  scoreResult.value = null
  try {
    const payload = { modelId: m.id, board: 'all', boards: boards.value, poolSize: Number(poolSize.value), assetClass: assetClass.value }
    if (adjustPanel.value === m.id) {
      const adj = adjustPayload()
      if (Object.keys(adj.featureWeights).length || adj.threshold !== null) payload.adjust = adj
    }
    scoreResult.value = await longTask('/ml/score', payload)
    toast(`打分完成，共 ${scoreResult.value.length} 只`)
  } catch (e) { toast(e.message) }
  finally { scoring.value = '' }
}

// 用模型预测分做分层回测（ML→回测闭环，走 jobs 异步；btStart/btEnd 限定验证区间）
async function runMLBacktest(m) {
  btLoading.value = m.id
  btResult.value = null
  try {
    const { jobId } = await api.post('/jobs', { kind: 'ml-backtest', config: {
      modelId: m.id, board: 'all', boards: boards.value, poolSize: Number(poolSize.value),
      n: Number(n.value), hist: Number(hist.value), assetClass: assetClass.value,
      startDate: btStart.value || null, endDate: btEnd.value || null,
    }})
    btResult.value = await pollJob(jobId)
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
const manualWeights = reactive({})
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
  return {
    grid: { left: 50, right: 20, top: 20, bottom: 30 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: pts.map(p => p.date) },
    yAxis: { type: 'value', axisLabel: { formatter: v => (v * 100).toFixed(1) + '%' } },
    series: [{
      name: '多空累计', type: 'line', smooth: true,
      data: pts.map(p => Number((p.cum * 100).toFixed(3))),
      itemStyle: { color: '#4f8cff' }, areaStyle: { color: 'rgba(79,140,255,.12)' },
    }],
  }
})

onMounted(() => { loadModels(); loadSectors(); loadManualFeatures() })
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
        <span class="hint">留空=用最近 hist 天全部样本；限定后仅用该区间样本训练/评估</span>
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
        <button class="btn-ghost" :disabled="loading" @click="runEvaluate">{{ loading ? '评估中…' : '评估(CV)' }}</button>
        <button class="btn-primary" :disabled="training" @click="runTrain">{{ training ? '训练中…' : '训练并落盘' }}</button>
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
        <div class="field grow"><label>规则说明（可选）</label><input v-model="manualRule" placeholder="如：动量>0.1 且 RSI<30 买入" /></div>
        <button class="btn-primary" :disabled="creatingManual" @click="saveManualModel">{{ creatingManual ? '创建中…' : '创建人造模型' }}</button>
      </div>
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
        <span class="hint">「ML回测」的验证区间（分时段验证），留空=整个历史</span>
      </div>
      <div v-if="!models.length" class="empty-hint">暂无模型，点击上方「训练并落盘」「创建人造模型」或导入模型文件</div>
      <table v-else class="data-table">
        <thead><tr><th>模型ID</th><th>类型</th><th>因子数</th><th>可回测</th><th>操作</th></tr></thead>
        <tbody>
          <tr v-for="m in models" :key="m.id" :class="{ 'row-disabled': m.computable === false }">
            <td>
              {{ m.id }}
              <span v-if="m.computable === false" class="badge-warn" :title="'未知特征: ' + (m.unknownFeatures||[]).join(', ')">因子不可计算</span>
            </td>
            <td class="muted">{{ m.modelType || '?' }}</td>
            <td class="muted">{{ m.nFeatures ?? '?' }}</td>
            <td><span :class="m.computable === false ? 'tag-fail' : 'tag-ok'">{{ m.computable === false ? '否' : '是' }}</span></td>
            <td>
              <button class="btn-ghost sm" :disabled="scoring===m.id || !m.computable" :title="!m.computable ? '模型因子不可从K线计算，无法回测' : ''" @click="runScore(m)">{{ scoring===m.id ? '打分中…' : '打分选股' }}</button>
              <button class="btn-ghost sm" :disabled="btLoading===m.id || !m.computable" :title="!m.computable ? '模型因子不可从K线计算，无法回测' : ''" @click="runMLBacktest(m)">{{ btLoading===m.id ? '回测中…' : 'ML回测' }}</button>
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
          <span v-if="lastAdjustId" class="hint">已保存 adjustId：{{ lastAdjustId }}（打分/回测自动生效）</span>
        </div>
        <div v-if="adjustNote" class="adj-note">{{ adjustNote }}</div>
      </div>
    </div>

    <div v-if="scoreResult" class="card">
      <div class="card-head"><h3>模型截面打分</h3><span class="hint">最新截面预测分排序（Top 30）</span></div>
      <table class="data-table">
        <thead><tr><th>排名</th><th>代码</th><th>名称</th><th>预测分</th></tr></thead>
        <tbody>
          <tr v-for="(r,i) in scoreResult.slice(0,30)" :key="r.code">
            <td>{{ i + 1 }}</td><td>{{ r.code }}</td><td>{{ r.name }}</td>
            <td class="up">{{ fmt(r.score) }}</td>
          </tr>
        </tbody>
      </table>
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
