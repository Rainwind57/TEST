<script setup>
import { ref, computed, onMounted } from 'vue'
import api from '../api/client'
import { useToast } from '../stores/toast'
import EChart from '../components/EChart.vue'

const { toast } = useToast()

const BOARD_OPTIONS = [
  { value: 'all', label: '全部A股' },
  { value: 'sh_main', label: '沪市主板' },
  { value: 'sz_main', label: '深市主板' },
  { value: 'gem', label: '创业板' },
  { value: 'star', label: '科创板' },
  { value: 'bse', label: '北交所' }
]

const board = ref('all')
const poolSize = ref(60)
const factorKey = ref('momentum')
const groups = ref(5)
const days = ref(5)
const hist = ref(180)
const loading = ref(false)
const result = ref(null)
const factorOptions = ref([])

async function loadFactorOptions() {
  const data = await api.get('/select/factors')
  factorOptions.value = data.filter(f => f.kline)
}

async function runBacktest() {
  loading.value = true
  try {
    const data = await api.post('/select/backtest', {
      board: board.value, poolSize: Number(poolSize.value), factor: factorKey.value,
      groups: Number(groups.value), n: Number(days.value), hist: Number(hist.value)
    })
    result.value = data
  } catch (e) {
    toast(e.message)
    result.value = null
  } finally {
    loading.value = false
  }
}

const groupBarOption = computed(() => {
  if (!result.value) return {}
  const data = result.value.groupSummary
  return {
    backgroundColor: 'transparent',
    grid: { left: 60, right: 30, top: 30, bottom: 40 },
    tooltip: { trigger: 'axis' },
    xAxis: {
      type: 'category', data: data.map(d => `G${d.group}`),
      axisLine: { lineStyle: { color: '#2a3354' } }, axisLabel: { color: '#7c89a8' }
    },
    yAxis: {
      type: 'value', name: `平均未来${result.value.n}日收益率`, nameTextStyle: { color: '#7c89a8' },
      axisLine: { lineStyle: { color: '#2a3354' } }, axisLabel: { color: '#7c89a8', formatter: v => (v * 100).toFixed(1) + '%' },
      splitLine: { lineStyle: { color: '#1c2238' } }
    },
    series: [{
      type: 'bar', data: data.map(d => d.avgReturn),
      itemStyle: { color: (p) => (p.value >= 0 ? '#ff4d4f' : '#21c08b') }
    }]
  }
})

const icLineOption = computed(() => {
  if (!result.value) return {}
  const s = result.value.icSeries
  return {
    backgroundColor: 'transparent',
    grid: { left: 60, right: 30, top: 40, bottom: 60 },
    tooltip: { trigger: 'axis' },
    legend: { textStyle: { color: '#e6ebf5' }, top: 0 },
    xAxis: {
      type: 'category', data: s.map(d => d.date),
      axisLine: { lineStyle: { color: '#2a3354' } }, axisLabel: { color: '#7c89a8', rotate: 45 }
    },
    yAxis: {
      type: 'value', axisLine: { lineStyle: { color: '#2a3354' } }, axisLabel: { color: '#7c89a8' },
      splitLine: { lineStyle: { color: '#1c2238' } }
    },
    series: [
      { name: 'IC', type: 'line', data: s.map(d => d.ic), showSymbol: false, lineStyle: { color: '#4f8cff', width: 2 } },
      { name: 'RankIC', type: 'line', data: s.map(d => d.rankIc), showSymbol: false, lineStyle: { color: '#6c5ce7', width: 2 } }
    ]
  }
})

const longShortOption = computed(() => {
  if (!result.value) return {}
  const s = result.value.longShort
  return {
    backgroundColor: 'transparent',
    grid: { left: 60, right: 30, top: 30, bottom: 60 },
    tooltip: { trigger: 'axis' },
    xAxis: {
      type: 'category', data: s.map(d => d.date),
      axisLine: { lineStyle: { color: '#2a3354' } }, axisLabel: { color: '#7c89a8', rotate: 45 }
    },
    yAxis: {
      type: 'value', axisLine: { lineStyle: { color: '#2a3354' } },
      axisLabel: { color: '#7c89a8', formatter: v => (v * 100).toFixed(0) + '%' },
      splitLine: { lineStyle: { color: '#1c2238' } }
    },
    series: [{
      name: '多空累计收益', type: 'line', data: s.map(d => d.cum), showSymbol: false,
      lineStyle: { color: '#ff4d4f', width: 2 }, areaStyle: { color: 'rgba(255,77,79,.12)' }
    }]
  }
})

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
      <div class="field"><label>候选池规模</label><input v-model="poolSize" type="number" min="20" max="300" /></div>
      <div class="field"><label>因子</label>
        <select v-model="factorKey">
          <option v-for="f in factorOptions" :key="f.key" :value="f.key">{{ f.label }}</option>
        </select>
      </div>
      <div class="field"><label>分组数</label><input v-model="groups" type="number" min="2" max="10" /></div>
      <div class="field"><label>持有天数 N</label><input v-model="days" type="number" min="1" max="30" /></div>
      <div class="field"><label>历史长度(日)</label><input v-model="hist" type="number" min="60" max="360" /></div>
      <button class="btn-primary" :disabled="loading" @click="runBacktest">{{ loading ? '回测中…' : '运行分层回测' }}</button>
      <span class="hint">对候选池股票做滚动截面分组：每 N 个交易日按因子值将全部股票分成若干组，统计各组未来 N 日平均收益、逐期 IC/RankIC 与多空组合累计收益</span>
    </div>

    <div v-if="!result" class="empty-hint">设置参数后点击「运行分层回测」</div>
    <template v-else>
      <div class="stat-cards">
        <div class="stat-card"><div class="label">有效股票数</div><div class="value">{{ result.effectiveStocks }}</div></div>
        <div class="stat-card"><div class="label">调仓期数</div><div class="value">{{ result.rebalanceCount }}</div></div>
        <div class="stat-card"><div class="label">平均 IC</div><div class="value" :class="result.meanIc >= 0 ? 'up' : 'down'">{{ result.meanIc.toFixed(4) }}</div></div>
        <div class="stat-card"><div class="label">IC 胜率</div><div class="value">{{ (result.icWinRate * 100).toFixed(1) }}%</div></div>
      </div>

      <div class="card chart-card">
        <div class="card-head"><h3>分层收益（组1=因子值最低，组N=因子值最高）</h3></div>
        <EChart :option="groupBarOption" height="320px" />
      </div>
      <div class="card chart-card">
        <div class="card-head"><h3>IC / RankIC 时序列</h3></div>
        <EChart :option="icLineOption" height="320px" />
      </div>
      <div class="card chart-card">
        <div class="card-head"><h3>多空组合（最高组-最低组）累计收益曲线</h3></div>
        <EChart :option="longShortOption" height="320px" />
      </div>
    </template>
  </div>
</template>
