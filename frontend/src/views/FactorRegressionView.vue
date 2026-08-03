<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import api from '../api/client'
import { useToast } from '../stores/toast'
import { useResearchStore } from '../stores/research'
import EChart from '../components/EChart.vue'

const { toast } = useToast()
const router = useRouter()
const research = useResearchStore()

const BOARD_OPTIONS = [
  { value: 'all', label: '全部A股' },
  { value: 'sh_main', label: '沪市主板' },
  { value: 'sz_main', label: '深市主板' },
  { value: 'gem', label: '创业板' },
  { value: 'star', label: '科创板' },
  { value: 'bse', label: '北交所' }
]

const board = ref('all')
const poolSize = ref(80)
const days = ref(5)
const hist = ref(200)
const loading = ref(false)
const result = ref(null)
const factorOptions = ref([])
const selectedKeys = ref(['momentum', 'ma_dev'])

async function loadFactorOptions() {
  const data = await api.get('/select/factors')
  factorOptions.value = data.filter(f => f.kline)
}

function toggleKey(key) {
  const idx = selectedKeys.value.indexOf(key)
  if (idx >= 0) selectedKeys.value.splice(idx, 1)
  else selectedKeys.value.push(key)
}

async function runRegression() {
  if (selectedKeys.value.length < 2) { toast('请至少选择2个因子'); return }
  loading.value = true
  try {
    const data = await api.post('/select/factor-regression', {
      board: board.value, poolSize: Number(poolSize.value), factors: selectedKeys.value,
      n: Number(days.value), hist: Number(hist.value)
    })
    result.value = data
  } catch (e) {
    toast(e.message)
    result.value = null
  } finally {
    loading.value = false
  }
}

const returnLineOption = computed(() => {
  if (!result.value) return {}
  const p = result.value.periods
  return {
    backgroundColor: 'transparent',
    grid: { left: 60, right: 30, top: 40, bottom: 60 },
    tooltip: { trigger: 'axis' },
    legend: { textStyle: { color: '#5b6675' }, top: 0 },
    xAxis: {
      type: 'category', data: p.map(d => d.date),
      axisLine: { lineStyle: { color: '#d7dce8' } }, axisLabel: { color: '#8a94a6', rotate: 45 }
    },
    yAxis: {
      type: 'value', axisLine: { lineStyle: { color: '#d7dce8' } }, axisLabel: { color: '#8a94a6' },
      splitLine: { lineStyle: { color: '#e9edf5' } }
    },
    series: result.value.keys.map((k, idx) => ({
      name: result.value.summary[idx].label, type: 'line', showSymbol: false,
      data: p.map(d => d.coefs[k])
    }))
  }
})

// 用首个显著因子带到主回测页做分层回测（打通多因子回归→回测）
function gotoBacktest() {
  const sig = result.value?.summary?.find(s => Math.abs(s.tStat) >= 2) || result.value?.summary?.[0]
  const factor = sig?.key || selectedKeys.value[0]
  research.setOptimalParams({ factor, board: board.value, poolSize: Number(poolSize.value), n: Number(days.value), hist: Number(hist.value) })
  router.push('/backtest')
}

onMounted(loadFactorOptions)
</script>

<template>
  <div>
    <div class="panel-toolbar">
      <div class="field"><label>板块</label>
        <select v-model="board">
          <option v-for="b in BOARD_OPTIONS" :key="b.value" :value="b.value">{{ b.label }}</option>
        </select>
      </div>
      <div class="field"><label>候选池规模</label><input v-model="poolSize" type="number" min="30" max="300" /></div>
      <div class="field"><label>持有天数 N</label><input v-model="days" type="number" min="1" max="30" /></div>
      <div class="field"><label>历史长度(日)</label><input v-model="hist" type="number" min="60" max="360" /></div>
      <button class="btn-primary" :disabled="loading" @click="runRegression">{{ loading ? '计算中…' : '运行多因子回归' }}</button>
      <button class="btn-ghost" v-if="result" @click="gotoBacktest">用显著因子回测</button>
    </div>
    <div class="hint" style="margin-bottom:12px">
      Fama-MacBeth 风格截面回归：每 N 个交易日用截面因子暴露（z-score 标准化）对未来 N 日收益做多元线性回归，
      逐期因子收益汇总为时序，均值/t 统计量反映因子长期是否显著有效（|t|≥2 通常视为显著）
    </div>

    <div class="card" style="margin-bottom:16px">
      <div class="card-head"><h3>选择因子（至少2个）</h3></div>
      <div class="chip-grid">
        <label v-for="f in factorOptions" :key="f.key" class="chip" :class="{ active: selectedKeys.includes(f.key) }">
          <input type="checkbox" :checked="selectedKeys.includes(f.key)" @change="toggleKey(f.key)" />
          {{ f.label }}
        </label>
      </div>
    </div>

    <div v-if="!result" class="empty-hint">设置参数后点击「运行多因子回归」</div>
    <template v-else>
      <div class="stat-cards">
        <div class="stat-card"><div class="label">有效股票数</div><div class="value">{{ result.effectiveStocks }}</div></div>
        <div class="stat-card"><div class="label">调仓期数</div><div class="value">{{ result.rebalanceCount }}</div></div>
        <div class="stat-card"><div class="label">平均 R²</div><div class="value">{{ result.meanR2.toFixed(4) }}</div></div>
      </div>

      <div class="card" style="margin-bottom:16px">
        <div class="card-head"><h3>因子收益归因（截面回归系数时序统计）</h3></div>
        <table>
          <thead>
            <tr><th>因子</th><th>平均因子收益</th><th>t 统计量</th><th>胜率</th><th>显著性</th></tr>
          </thead>
          <tbody>
            <tr v-for="s in result.summary" :key="s.key">
              <td class="td-name">{{ s.label }}</td>
              <td :class="s.meanReturn >= 0 ? 'up' : 'down'">{{ (s.meanReturn * 100).toFixed(3) }}%</td>
              <td :class="Math.abs(s.tStat) >= 2 ? 'up' : ''">{{ s.tStat.toFixed(3) }}</td>
              <td>{{ (s.positiveRate * 100).toFixed(1) }}%</td>
              <td>{{ Math.abs(s.tStat) >= 2 ? '显著' : '不显著' }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="card chart-card">
        <div class="card-head"><h3>各因子逐期收益时序列</h3></div>
        <EChart :option="returnLineOption" height="340px" />
      </div>
    </template>
  </div>
</template>

<style scoped>
.chip-grid { display: flex; flex-wrap: wrap; gap: 10px; padding: 16px 22px; }
.chip {
  display: flex; align-items: center; gap: 6px;
  padding: 6px 12px; border-radius: 8px; border: 1px solid var(--border);
  color: var(--text-dim); font-size: 12px; cursor: pointer;
}
.chip.active { border-color: var(--accent); color: var(--text); background: rgba(79,140,255,.12); }
</style>
