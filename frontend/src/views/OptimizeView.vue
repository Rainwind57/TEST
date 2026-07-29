<script setup>
import { ref, computed, onMounted } from 'vue'
import api from '../api/client'
import { useToast } from '../stores/toast'
import EChart from '../components/EChart.vue'

const { toast } = useToast()

const board = ref('all')
const poolSize = ref(60)
const factor = ref('momentum')
const groups = ref(5)
const n = ref(5)
const hist = ref(180)
const nTrials = ref(30)
const benchmark = ref('none')

const loading = ref(false)
const result = ref(null)
const factorOptions = ref([])

async function loadFactors() {
  try { factorOptions.value = (await api.get('/select/factors')).filter(f => f.kline) }
  catch (e) { toast(e.message) }
}

async function run() {
  loading.value = true
  try {
    const cfg = {
      board: board.value, poolSize: Number(poolSize.value), factor: factor.value,
      groups: Number(groups.value), n: Number(n.value), hist: Number(hist.value),
      benchmark: benchmark.value, nTrials: Number(nTrials.value),
    }
    result.value = await api.post('/optimize/backtest', cfg)
    toast(`寻优完成，${result.value.nTrials} 次试验`)
  } catch (e) { toast(e.message) }
  finally { loading.value = false }
}

async function saveStrategy() {
  if (!result.value) return
  const name = prompt('策略名称', `${factor.value}_opt_${nTrials.value}trials`)
  if (!name) return
  try {
    await api.post('/optimize/save-strategy', {
      name,
      baseConfig: { board: board.value, factor: factor.value, hist: Number(hist.value),
        benchmark: benchmark.value, commissionRate: 0.00025, stampDuty: 0.001,
        slippage: 0.001, applyCost: true },
      bestParams: result.value.bestParams,
    })
    toast('已回写为策略')
  } catch (e) { toast(e.message) }
}

const trialOption = computed(() => {
  if (!result.value?.trials?.length) return {}
  const data = result.value.trials
  return {
    tooltip: { trigger: 'axis' },
    grid: { left: 50, right: 20, top: 30, bottom: 30 },
    xAxis: { type: 'category', data: data.map(t => t.number) },
    yAxis: { type: 'value', name: 'Sharpe' },
    series: [{ type: 'line', data: data.map(t => Number(t.value).toFixed(3)),
      itemStyle: { color: '#4f8cff' } }]
  }
})

const fmt = v => v == null ? '-' : Number(v).toFixed(3)
const fmtPct = v => v == null ? '-' : (Number(v) * 100).toFixed(2) + '%'

onMounted(loadFactors)
</script>

<template>
  <div>
    <div class="card">
      <div class="card-head"><h3>参数寻优 · Optuna 贝叶斯搜索</h3><span class="hint">Walk-Forward：IS 区间调参，OOS 区间验证</span></div>
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
        <div class="field"><label>因子</label>
          <select v-model="factor">
            <option v-for="f in factorOptions" :key="f.key" :value="f.key">{{ f.label }}</option>
          </select>
        </div>
        <div class="field"><label>历史长度</label><input v-model="hist" type="number" /></div>
        <div class="field"><label>试验数</label><input v-model="nTrials" type="number" /></div>
        <div class="field"><label>基准</label>
          <select v-model="benchmark">
            <option value="none">无</option>
            <option value="hs300">沪深300</option>
            <option value="zz500">中证500</option>
          </select>
        </div>
      </div>
      <div class="panel-toolbar" style="margin-top:10px">
        <button class="btn-primary" :disabled="loading" @click="run">{{ loading ? '寻优中…' : '开始寻优' }}</button>
        <button class="btn-ghost" v-if="result" @click="saveStrategy">回写为策略</button>
      </div>
    </div>

    <div v-if="result" class="card">
      <h3>寻优结果</h3>
      <div class="kpi-row">
        <div class="kpi"><div class="n">{{ result.bestParams.poolSize }}</div><div class="l">最优池规模</div></div>
        <div class="kpi"><div class="n">{{ result.bestParams.groups }}</div><div class="l">分组数</div></div>
        <div class="kpi"><div class="n">{{ result.bestParams.n }}</div><div class="l">持有期N</div></div>
        <div class="kpi"><div class="n">{{ result.nTrials }}</div><div class="l">试验次数</div></div>
      </div>
      <table class="data-table">
        <thead><tr><th>区间</th><th>Sharpe</th><th>年化收益</th><th>最大回撤</th><th>累计收益</th><th>胜率</th></tr></thead>
        <tbody>
          <tr><td>IS(调参)</td>
            <td>{{ fmt(result.isMetrics?.sharpe) }}</td>
            <td class="up">{{ fmtPct(result.isMetrics?.annualizedReturn) }}</td>
            <td class="down">{{ fmtPct(result.isMetrics?.maxDrawdown) }}</td>
            <td>{{ fmtPct(result.isMetrics?.cumulativeReturn) }}</td>
            <td>{{ fmtPct(result.isMetrics?.winRate) }}</td>
          </tr>
          <tr><td>OOS(验证)</td>
            <td>{{ fmt(result.oosMetrics?.sharpe) }}</td>
            <td class="up">{{ fmtPct(result.oosMetrics?.annualizedReturn) }}</td>
            <td class="down">{{ fmtPct(result.oosMetrics?.maxDrawdown) }}</td>
            <td>{{ fmtPct(result.oosMetrics?.cumulativeReturn) }}</td>
            <td>{{ fmtPct(result.oosMetrics?.winRate) }}</td>
          </tr>
        </tbody>
      </table>
      <div v-if="result.oosMetrics?.error" class="warn-box">OOS 失败：{{ result.oosMetrics.error }}</div>
    </div>

    <div v-if="result?.trials?.length" class="card">
      <h3>试验收敛曲线</h3>
      <EChart :option="trialOption" style="height:280px" />
    </div>
  </div>
</template>

<style scoped>
.data-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
.data-table th, .data-table td { border: 1px solid var(--border); padding: 9px 11px; text-align: left; }
.data-table th { background: var(--card-2); color: var(--text-dim); font-weight: 600; }
.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px; margin: 14px 0; }
.kpi { background: var(--card-2); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }
.kpi .n { font-size: 20px; font-weight: 700; color: var(--accent); }
.kpi .l { color: var(--text-mute); font-size: 12px; margin-top: 4px; }
.hint { color: var(--text-mute); font-size: 12px; font-weight: 400; }
.card-head { display: flex; justify-content: space-between; align-items: baseline; }
.warn-box { background: rgba(255,180,84,.08); border: 1px solid #ffb45444; border-radius: 8px; padding: 10px; margin-top: 10px; font-size: 13px; color: #ffd9a0; }
</style>
