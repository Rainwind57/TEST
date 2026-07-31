<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import api, { longTask, downloadFile } from '../api/client'
import { useToast } from '../stores/toast'
import { useResearchStore } from '../stores/research'
import EChart from '../components/EChart.vue'

const { toast } = useToast()
const research = useResearchStore()
const router = useRouter()

const board = ref('all')
const poolSize = ref(80)
const n = ref(5)
const hist = ref(240)
const modelType = ref('gbdt')
const nSplits = ref(5)
const gap = ref(5)
const useSnapshot = ref(false)

const loading = ref(false)
const training = ref(false)
const trainMsg = ref('')
const result = ref(null)
const models = ref([])

const scoring = ref('')
const scoreResult = ref(null)
const btLoading = ref('')
const btResult = ref(null)

function evalConfig() {
  return {
    board: board.value, poolSize: Number(poolSize.value), n: Number(n.value),
    hist: Number(hist.value), modelType: modelType.value,
    nSplits: Number(nSplits.value), gap: Number(gap.value),
    useSnapshot: useSnapshot.value,
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

// 用模型对候选池最新截面打分（ML→选股闭环）
async function runScore(m) {
  scoring.value = m.id
  scoreResult.value = null
  try {
    scoreResult.value = await longTask('/ml/score', {
      modelId: m.id, board: board.value, poolSize: Number(poolSize.value),
    })
    toast(`打分完成，共 ${scoreResult.value.length} 只`)
  } catch (e) { toast(e.message) }
  finally { scoring.value = '' }
}

// 用模型预测分做分层回测（ML→回测闭环，走 jobs 异步）
async function runMLBacktest(m) {
  btLoading.value = m.id
  btResult.value = null
  try {
    const { jobId } = await api.post('/jobs', { kind: 'ml-backtest', config: {
      modelId: m.id, board: board.value, poolSize: Number(poolSize.value),
      n: Number(n.value), hist: Number(hist.value),
    }})
    btResult.value = await pollJob(jobId)
    toast(`ML 回测完成，调仓 ${btResult.value.rebalanceCount} 次`)
  } catch (e) { toast(e.message) }
  finally { btLoading.value = '' }
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
    await downloadFile('/reports/backtest', payload, `ml_backtest.${fmt === 'html' ? 'html' : 'xlsx'}`)
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

onMounted(loadModels)
onUnmounted(() => { pollActive = false })
</script>

<template>
  <div>
    <div class="card">
      <div class="card-head"><h3>机器学习 · 因子收益预测</h3><span class="hint">GBDT + 时序 Walk-Forward CV，防前视偏差</span></div>
      <div class="panel-toolbar">
        <div class="field"><label>板块</label>
          <select v-model="board">
            <option value="all">全部A股</option>
            <option value="sh_main">沪市主板</option>
            <option value="sz_main">深市主板</option>
            <option value="gem">创业板</option>
            <option value="star">科创板</option>
          </select>
        </div>
        <div class="field"><label>候选池</label><input v-model="poolSize" type="number" /></div>
        <div class="field"><label>预测期N(日)</label><input v-model="n" type="number" /></div>
        <div class="field"><label>历史长度</label><input v-model="hist" type="number" /></div>
        <div class="field"><label>CV折数</label><input v-model="nSplits" type="number" /></div>
        <div class="field"><label>Gap</label><input v-model="gap" type="number" /></div>
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
      <div class="card-head"><h3>已训练模型</h3><span class="hint">joblib 落盘，可用 joblib.load 复用</span></div>
      <div v-if="!models.length" class="empty-hint">暂无模型，点击上方「训练并落盘」</div>
      <table v-else class="data-table">
        <thead><tr><th>模型ID</th><th>文件</th><th>操作</th></tr></thead>
        <tbody>
          <tr v-for="m in models" :key="m.id">
            <td>{{ m.id }}</td><td class="muted">{{ m.file }}</td>
            <td>
              <button class="btn-ghost sm" :disabled="scoring===m.id" @click="runScore(m)">{{ scoring===m.id ? '打分中…' : '打分选股' }}</button>
              <button class="btn-ghost sm" :disabled="btLoading===m.id" @click="runMLBacktest(m)">{{ btLoading===m.id ? '回测中…' : 'ML回测' }}</button>
              <button class="btn-ghost sm" @click="gotoBacktest(m)">去主回测</button>
              <button class="btn-ghost sm danger" @click="deleteModel(m.id)">删除</button>
            </td>
          </tr>
        </tbody>
      </table>
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
.btn-ghost.sm { padding: 4px 10px; font-size: 12px; }
</style>
