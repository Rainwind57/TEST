<script setup>
import { ref, computed, onMounted } from 'vue'
import api from '../api/client'
import { useToast } from '../stores/toast'
import EChart from '../components/EChart.vue'

const { toast } = useToast()

const board = ref('all')
const poolSize = ref(80)
const n = ref(5)
const hist = ref(240)
const modelType = ref('gbdt')
const nSplits = ref(5)
const gap = ref(5)

const loading = ref(false)
const training = ref(false)
const result = ref(null)
const models = ref([])

async function runEvaluate() {
  loading.value = true
  try {
    result.value = await api.post('/ml/evaluate', {
      board: board.value, poolSize: Number(poolSize.value), n: Number(n.value),
      hist: Number(hist.value), modelType: modelType.value,
      nSplits: Number(nSplits.value), gap: Number(gap.value),
    })
    toast(`评估完成，OOS IC=${(result.value.oosIc || 0).toFixed(3)}`)
  } catch (e) { toast(e.message) }
  finally { loading.value = false }
}

async function runTrain() {
  training.value = true
  try {
    const res = await api.post('/ml/train', {
      board: board.value, poolSize: Number(poolSize.value), n: Number(n.value),
      hist: Number(hist.value), modelType: modelType.value,
      nSplits: Number(nSplits.value), gap: Number(gap.value),
    })
    result.value = res.evaluation
    models.value = await api.get('/ml/models')
    toast(`训练完成，模型 ${res.model.id} 已落盘`)
  } catch (e) { toast(e.message) }
  finally { training.value = false }
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

onMounted(loadModels)
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
            <td><button class="btn-ghost sm danger" @click="deleteModel(m.id)">删除</button></td>
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
.btn-ghost.sm { padding: 4px 10px; font-size: 12px; }
</style>
