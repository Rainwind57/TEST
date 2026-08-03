<script setup>
import { ref, computed, onMounted } from 'vue'
import api from '../api/client'
import { useToast } from '../stores/toast'
import { useWatchlistStore } from '../stores/watchlist'
import EChart from '../components/EChart.vue'
import { fmtNum, fmtPct, stripPrefix } from '../utils/format'

const { toast } = useToast()
const watchlist = useWatchlistStore()

// ---- 期货行情 ----
const FUTURE_PRESETS = ['IF2608', 'rb2610', 'au2612', 'm2609', 'SR609']
const futureInput = ref('')
const futureQuotes = ref({})
const futureLoading = ref(false)
const activeFuture = ref(null)
const futureKline = ref([])
const futureKlineLoading = ref(false)

async function loadFutureQuotes() {
  const raw = futureInput.value.trim() || FUTURE_PRESETS.join(',')
  futureInput.value = raw
  futureLoading.value = true
  try {
    futureQuotes.value = await api.get('/data/futures/quotes', { params: { codes: raw } })
    if (!Object.keys(futureQuotes.value).length) toast('未获取到行情，请检查合约代码')
  } catch (e) { toast(e.message) }
  finally { futureLoading.value = false }
}

async function loadFutureKline(code) {
  activeFuture.value = code
  futureKlineLoading.value = true
  try {
    const res = await api.get('/data/futures/kline/' + code, { params: { days: 90 } })
    futureKline.value = res.rows || []
  } catch (e) { futureKline.value = []; toast(e.message) }
  finally { futureKlineLoading.value = false }
}

const futureKlineOption = computed(() => {
  const rows = futureKline.value
  return {
    backgroundColor: 'transparent',
    grid: { left: 50, right: 20, top: 20, bottom: 30 },
    xAxis: { type: 'category', data: rows.map(k => k.date), axisLine: { lineStyle: { color: '#d7dce8' } }, axisLabel: { color: '#8a94a6' } },
    yAxis: { scale: true, axisLine: { lineStyle: { color: '#d7dce8' } }, axisLabel: { color: '#8a94a6' }, splitLine: { lineStyle: { color: '#e9edf5' } } },
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    series: [{
      type: 'candlestick', data: rows.map(k => [k.open, k.close, k.low, k.high]),
      itemStyle: { color: '#ff4d4f', color0: '#21c08b', borderColor: '#ff4d4f', borderColor0: '#21c08b' }
    }]
  }
})

// ---- 宏观指标 ----
const MACRO_LIST = [
  { key: 'CPI', label: 'CPI当月同比(%)' },
  { key: 'PPI', label: 'PPI当月同比(%)' },
  { key: 'PMI', label: '制造业PMI' },
  { key: 'M2', label: 'M2同比(%)' }
]
const macroData = ref({})
async function loadMacro() {
  const entries = await Promise.all(MACRO_LIST.map(async m => {
    try { return [m.key, await api.get('/data/macro/' + m.key)] }
    catch (e) { return [m.key, null] }
  }))
  macroData.value = Object.fromEntries(entries)
}

// ---- 指数成分 ----
const INDEX_LIST = [
  { code: 'sh000300', label: '沪深300' },
  { code: 'sh000905', label: '中证500' }
]
const indexConstituents = ref(null)
const indexLoading = ref(false)
async function loadIndex(code) {
  indexLoading.value = true
  try {
    indexConstituents.value = await api.get('/data/index-constituents/' + code)
  } catch (e) { indexConstituents.value = null; toast(e.message) }
  finally { indexLoading.value = false }
}

async function addConstituentsToWatchlist() {
  if (!indexConstituents.value) return
  let added = 0
  for (const code of indexConstituents.value.codes.slice(0, 50)) {
    try { await watchlist.addCode(code); added++ } catch (e) { /* 已存在忽略 */ }
  }
  toast(`已加入自选 ${added} 只（前50只）`)
}

onMounted(() => { loadFutureQuotes(); loadMacro() })
</script>

<template>
  <div>
    <!-- 期货行情 -->
    <div class="card mb-24">
      <div class="card-head"><h2>期货行情</h2><span class="hint">支持中金所（IF/IH/IC/T…）与商品合约，如 IF2608,rb2610,au2612</span></div>
      <div class="search-bar">
        <input v-model="futureInput" placeholder="输入合约代码，逗号分隔，如 IF2608,rb2610,au2612" @keydown.enter="loadFutureQuotes" />
        <button class="btn-primary" :disabled="futureLoading" @click="loadFutureQuotes">{{ futureLoading ? '加载中…' : '查询' }}</button>
      </div>
      <table>
        <thead><tr><th>合约</th><th>名称</th><th>最新价</th><th>涨跌幅</th><th>今开</th><th>最高</th><th>最低</th><th>昨收</th><th>成交量(手)</th><th>K线</th></tr></thead>
        <tbody>
          <tr v-if="!Object.keys(futureQuotes).length" class="empty-row"><td colspan="10">暂无数据，输入合约代码查询</td></tr>
          <tr v-for="(q, code) in futureQuotes" :key="code">
            <td class="td-name">{{ stripPrefix(code) }}</td>
            <td>{{ q.name }}</td>
            <td :class="q.changePct > 0 ? 'up' : q.changePct < 0 ? 'down' : 'flat'">{{ fmtNum(q.price) }}</td>
            <td :class="q.changePct > 0 ? 'up' : q.changePct < 0 ? 'down' : 'flat'">{{ fmtPct(q.changePct) }}</td>
            <td>{{ fmtNum(q.open) }}</td>
            <td class="up">{{ fmtNum(q.high) }}</td>
            <td class="down">{{ fmtNum(q.low) }}</td>
            <td>{{ fmtNum(q.preClose) }}</td>
            <td>{{ fmtNum(q.volume) }}</td>
            <td><button class="btn-ghost sm" :disabled="futureKlineLoading" @click="loadFutureKline(code)">{{ activeFuture === code ? 'K线中…' : 'K线' }}</button></td>
          </tr>
        </tbody>
      </table>
      <div v-if="activeFuture && futureKline.length" class="chart-box">
        <div class="card-head"><h3>{{ activeFuture }} 日K线</h3></div>
        <EChart :option="futureKlineOption" height="300px" />
      </div>
    </div>

    <!-- 宏观指标 -->
    <div class="card mb-24">
      <div class="card-head"><h2>宏观指标</h2><span class="hint">东方财富数据中心，最新一期</span></div>
      <div class="macro-grid">
        <div class="macro-card" v-for="m in MACRO_LIST" :key="m.key">
          <div class="macro-label">{{ m.label }}</div>
          <div class="macro-value">{{ macroData[m.key] ? fmtNum(macroData[m.key].value) : '--' }}</div>
          <div class="macro-date">{{ macroData[m.key] ? (macroData[m.key].date || '').slice(0, 7) : '加载中…' }}</div>
        </div>
      </div>
    </div>

    <!-- 指数成分 -->
    <div class="card">
      <div class="card-head"><h2>指数成分股</h2><span class="hint">用于板块选股（hs300/zz500）与成分监控</span></div>
      <div class="panel-row">
        <button v-for="idx in INDEX_LIST" :key="idx.code" class="btn-ghost" :disabled="indexLoading" @click="loadIndex(idx.code)">{{ idx.label }}</button>
        <button class="btn-primary sm" style="margin-left:auto" :disabled="!indexConstituents" @click="addConstituentsToWatchlist">前50只加入自选</button>
      </div>
      <div v-if="indexConstituents" class="chips-box">
        <span class="hint">{{ indexConstituents.index }} 共 {{ indexConstituents.count }} 只（展示前 40）</span>
        <div class="chip-wrap">
          <span class="code-chip" v-for="c in indexConstituents.codes.slice(0, 40)" :key="c">{{ stripPrefix(c) }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chart-box { padding: 0 20px 20px; }
.macro-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 14px; padding: 4px 20px 20px; }
.macro-card { background: var(--bg-2); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.macro-label { font-size: 12px; color: var(--text-mute); }
.macro-value { font-size: 22px; font-weight: 700; color: var(--accent); margin: 6px 0 2px; }
.macro-date { font-size: 11px; color: var(--text-mute); }
.panel-row { display: flex; align-items: center; gap: 12px; padding: 16px 20px; flex-wrap: wrap; }
.chips-box { padding: 0 20px 20px; }
.chip-wrap { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.code-chip { background: var(--bg-2); border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; font-size: 12px; color: var(--text-dim); }
</style>
