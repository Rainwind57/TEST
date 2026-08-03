<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useWatchlistStore } from '../stores/watchlist'
import { useToast } from '../stores/toast'
import { useResearchStore } from '../stores/research'
import api from '../api/client'
import EChart from '../components/EChart.vue'

const store = useWatchlistStore()
const { toast } = useToast()
const router = useRouter()
const research = useResearchStore()

const factorKey = ref('ma_dev')
const methodKey = ref('ols')
const days = ref(5)
const hist = ref(150)
const loading = ref(false)
const result = ref(null)
const factorOptions = ref([])
const methodOptions = ref([])

async function loadOptions() {
  try {
    const [catalog, methods] = await Promise.all([
      api.get('/factors/catalog'),
      api.get('/regression/methods')
    ])
    factorOptions.value = catalog.filter(c => c.kline)
    methodOptions.value = methods
  } catch (e) {
    toast(e.message)
  }
}

async function runRegression() {
  if (!store.codes.length) { toast('请先在行情页添加自选股'); return }
  loading.value = true
  try {
    const data = await api.post('/regression', {
      codes: store.codes, factor: factorKey.value, method: methodKey.value,
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

const equation = computed(() => {
  if (!result.value) return ''
  const terms = result.value.coefs.map((c, i) => {
    const v = c.toFixed(4)
    if (i === 0) return v
    const xPart = i === 1 ? 'x' : `x^${i}`
    return `${c >= 0 ? '+' : ''}${v}${xPart}`
  })
  return `y = ${terms.join(' ')}`
})

const chartOption = computed(() => {
  if (!result.value) return {}
  return {
    backgroundColor: 'transparent',
    grid: { left: 60, right: 30, top: 30, bottom: 50 },
    tooltip: { trigger: 'item' },
    legend: { textStyle: { color: '#5b6675' }, bottom: 0 },
    xAxis: {
      type: 'value', name: result.value.factorLabel, nameTextStyle: { color: '#8a94a6' },
      axisLine: { lineStyle: { color: '#d7dce8' } }, axisLabel: { color: '#8a94a6' },
      splitLine: { lineStyle: { color: '#e9edf5' } }
    },
    yAxis: {
      type: 'value', name: `未来${result.value.n}日收益率`, nameTextStyle: { color: '#8a94a6' },
      axisLine: { lineStyle: { color: '#d7dce8' } }, axisLabel: { color: '#8a94a6' },
      splitLine: { lineStyle: { color: '#e9edf5' } }
    },
    series: [
      {
        name: '样本', type: 'scatter', symbolSize: 6,
        data: result.value.samples.map(s => [s.x, s.y]),
        itemStyle: { color: 'rgba(79,140,255,.5)' }
      },
      {
        name: `${result.value.methodLabel} (RankIC=${result.value.rankIc.toFixed(3)})`, type: 'line',
        data: result.value.line.map(p => [p.x, p.y]),
        showSymbol: false, lineStyle: { color: '#ff4d4f', width: 2 }
      }
    ]
  }
})

// 把当前因子带到主回测页做分层回测（打通回归→回测，旧版算完即死路）
function gotoBacktest() {
  research.setOptimalParams({ factor: factorKey.value, n: Number(days.value), hist: Number(hist.value) })
  router.push('/backtest')
}

onMounted(async () => {
  await loadOptions()
  if (!store.codes.length) await store.fetchWatchlist()
})
</script>

<template>
  <div>
    <div class="panel-toolbar">
      <div class="field">
        <label>因子</label>
        <select v-model="factorKey">
          <option v-for="f in factorOptions" :key="f.key" :value="f.key">{{ f.label }}</option>
        </select>
      </div>
      <div class="field">
        <label>回归方法</label>
        <select v-model="methodKey">
          <option v-for="m in methodOptions" :key="m.key" :value="m.key">{{ m.label }}</option>
        </select>
      </div>
      <div class="field"><label>预测天数 N</label><input v-model="days" type="number" min="1" max="30" /></div>
      <div class="field"><label>历史长度(日)</label><input v-model="hist" type="number" min="60" max="300" /></div>
      <button class="btn-primary" :disabled="loading" @click="runRegression">{{ loading ? '计算中…' : '运行回归' }}</button>
      <button class="btn-ghost" v-if="result" @click="gotoBacktest">用该因子回测</button>
      <span class="hint">对自选股历史行情做滚动窗口取样：因子值(t) vs 未来N日收益率(t→t+N)，样本合并后按所选方法拟合</span>
    </div>

    <div v-if="!result" class="empty-hint">选择因子、回归方法和参数后点击「运行回归」</div>
    <template v-else>
      <div class="stat-cards">
        <div class="stat-card"><div class="label">样本量</div><div class="value">{{ result.sampleSize }}</div></div>
        <div class="stat-card"><div class="label">回归方程</div><div class="value" style="font-size:13px">{{ equation }}</div></div>
        <div class="stat-card"><div class="label">R²</div><div class="value">{{ result.r2.toFixed(4) }}</div></div>
        <div class="stat-card"><div class="label">IC (Pearson)</div><div class="value" :class="result.ic >= 0 ? 'up' : 'down'">{{ result.ic.toFixed(4) }}</div></div>
      </div>
      <div class="card chart-card">
        <EChart :option="chartOption" height="380px" />
      </div>
    </template>
  </div>
</template>
