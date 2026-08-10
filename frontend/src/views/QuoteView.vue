<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useWatchlistStore } from '../stores/watchlist'
import { useToast } from '../stores/toast'
import api from '../api/client'
import EChart from '../components/EChart.vue'
import { normalizeCode, fmtNum, fmtVol, fmtAmount, fmtPct, trendCls, stripPrefix } from '../utils/format'

const store = useWatchlistStore()
const { toast } = useToast()
const codeInput = ref('')
const klineData = ref([])
const klineDays = ref(500)
const klineFreq = ref('D')
const timeshareData = ref([])
const timeshareWarning = ref('')
let timer = null
let refreshing = false

const SOURCE_META = {
  tencent: '腾讯行情（推荐） · qt.gtimg.cn',
  sina: '新浪财经 · 可能受防盗链限制',
  eastmoney: '东方财富 · JSON'
}

const activeQuote = computed(() => store.quotes[store.activeCode])

// ---- 技术指标（单选切换） ----
const activeIndicator = ref('')
const INDICATOR_COLORS = ['#00bcd4', '#ff9800', '#e91e63']

const chartHeight = computed(() => activeIndicator.value ? 620 : 520)

function calcMACD(closes) {
  const ema = (data, n) => { let k = 2/(n+1); let r = [data[0]]; for (let i = 1; i < data.length; i++) r.push(data[i]*k + r[i-1]*(1-k)); return r }
  const ema12 = ema(closes, 12), ema26 = ema(closes, 26)
  const dif = ema12.map((v, i) => v - ema26[i])
  const dea = ema(dif, 9)
  const bar = dif.map((v, i) => (v - dea[i]) * 2)
  return { dif, dea, bar }
}

function calcKDJ(highs, lows, closes, n = 9) {
  const k = [], d = [], j = []
  for (let i = 0; i < closes.length; i++) {
    if (i < n - 1) { k.push(null); d.push(null); j.push(null); continue }
    const h = Math.max(...highs.slice(i - n + 1, i + 1))
    const l = Math.min(...lows.slice(i - n + 1, i + 1))
    const rsv = l === h ? 50 : ((closes[i] - l) / (h - l)) * 100
    const pk = k[k.length - 1], pd = d[d.length - 1]
    const ck = pk != null ? pk * 2/3 + rsv * 1/3 : 50
    const cd = pd != null ? pd * 2/3 + ck * 1/3 : 50
    k.push(+ck.toFixed(2)); d.push(+cd.toFixed(2))
    j.push(+(3 * ck - 2 * cd).toFixed(2))
  }
  return { k, d, j }
}

function calcBOLL(closes, n = 20) {
  const mid = [], upper = [], lower = []
  for (let i = 0; i < closes.length; i++) {
    if (i < n - 1) { mid.push(null); upper.push(null); lower.push(null); continue }
    const slice = closes.slice(i - n + 1, i + 1)
    const ma = slice.reduce((a, b) => a + b, 0) / n
    const std = Math.sqrt(slice.reduce((s, v) => s + (v - ma) ** 2, 0) / n)
    mid.push(+ma.toFixed(3)); upper.push(+(ma + 2 * std).toFixed(3)); lower.push(+(ma - 2 * std).toFixed(3))
  }
  return { mid, upper, lower }
}

function toggleIndicator(name) {
  activeIndicator.value = activeIndicator.value === name ? '' : name
}

// ---- 多股对比 ----
const compareCodes = ref([])
const compareKlines = ref({})
const compareLoading = ref(false)

function toggleCompare(code) {
  const idx = compareCodes.value.indexOf(code)
  if (idx >= 0) { compareCodes.value.splice(idx, 1); delete compareKlines.value[code] }
  else if (compareCodes.value.length >= 5) { toast('最多对比5只股票') }
  else { compareCodes.value.push(code); loadCompareKline(code) }
}
async function loadCompareKline(code) {
  try {
    const res = await api.get('/kline', { params: { code, days: klineDays.value, freq: klineFreq.value } })
    compareKlines.value[code] = res.data
  } catch (e) { /* 静默 */ }
}
function clearCompare() { compareCodes.value = []; compareKlines.value = {} }

const compareOption = computed(() => {
  const cds = compareCodes.value.filter(c => compareKlines.value[c]?.length)
  if (!cds.length) return {}
  const firstData = compareKlines.value[cds[0]]
  const baseDates = firstData.map(k => k.date)
  const series = cds.map((code, ci) => {
    const data = compareKlines.value[code]
    const firstClose = data[0].close
    const returns = data.map(k => +((k.close / firstClose - 1) * 100).toFixed(2))
    const name = store.quotes[code]?.name || code
    return { name, type: 'line', data: returns, smooth: true, showSymbol: false,
      lineStyle: { width: 2, color: INDICATOR_COLORS[ci % INDICATOR_COLORS.length] },
      markLine: ci === 0 ? { silent: true, data: [{ yAxis: 0, label: { formatter: '基准0%' } }], lineStyle: { color: '#8a94a6', type: 'dashed' } } : undefined
    }
  })
  return {
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis', formatter: p => p.map(v => `${v.marker}${v.seriesName}：${v.value}%`).join('<br/>') },
    legend: { data: series.map(s => s.name), top: 0, textStyle: { fontSize: 11, color: '#8a94a6' } },
    grid: { left: 60, right: 20, top: 40, bottom: 30 },
    xAxis: { type: 'category', data: baseDates, axisLabel: { color: '#8a94a6', rotate: 30 }, axisLine: { lineStyle: { color: '#e9edf5' } } },
    yAxis: { axisLabel: { color: '#8a94a6', formatter: '{value}%' }, splitLine: { lineStyle: { color: '#e9edf5' } } },
    series,
  };
})

async function addCode() {
  const c = normalizeCode(codeInput.value)
  if (!c) { toast('代码格式错误，应为 6 位数字或 sh/sz/bj 前缀'); return }
  if (store.codes.includes(c)) { toast('已在自选列表中'); return }
  try {
    const r = await api.get('/stock/exists', { params: { code: c } })
    if (r.exists === false) { toast(`未找到该股票：${c}，请检查代码是否正确`); return }
    await store.addCode(c)
    codeInput.value = ''
    toast('已添加 ' + (r.name || c))
  } catch (e) { toast(e.message) }
}

async function removeCode(code) {
  try { await store.removeCode(code); toast('已删除') } catch (e) { toast(e.message) }
}

function selectCode(code) { store.activeCode = code }

async function loadKline() {
  if (!store.activeCode) { klineData.value = []; return }
  try {
    const res = await api.get('/kline', { params: { code: store.activeCode, days: klineDays.value, freq: klineFreq.value } })
    klineData.value = res.data
  } catch (e) { klineData.value = [] }
}

async function loadTimeshare() {
  if (!store.activeCode) { timeshareData.value = []; timeshareWarning.value = ''; return }
  try {
    const res = await api.get('/timeshare', { params: { code: store.activeCode } })
    timeshareData.value = res.data || []
    timeshareWarning.value = res.warning || ''
  } catch (e) { timeshareData.value = []; timeshareWarning.value = e.message || '分时数据请求失败' }
}

const klineOption = computed(() => {
  const data = klineData.value
  if (!data.length) return {}
  const dates = data.map(k => k.date)
  const values = data.map(k => [k.open, k.close, k.low, k.high])
  const closes = data.map(k => k.close)
  const highs = data.map(k => k.high)
  const lows = data.map(k => k.low)
  const volumes = data.map(k => k.volume)
  const ma = (n) => closes.map((_, i) => {
    if (i < n - 1) return null
    let s = 0; for (let j = i - n + 1; j <= i; j++) s += closes[j]
    return +(s / n).toFixed(3)
  })

  const indicator = activeIndicator.value
  const hasIndicators = !!indicator

  const grids = [
    { left: 60, right: 20, top: 40, bottom: hasIndicators ? 370 : 250 },
    { left: 60, right: 20, top: hasIndicators ? 258 : 280, bottom: hasIndicators ? 250 : 90 },
  ]
  const xAxes = [
    { type: 'category', data: dates, gridIndex: 0, axisLabel: { show: false }, axisTick: { show: false }, axisLine: { show: false } },
    { type: 'category', data: dates, gridIndex: 1, axisLabel: { show: !hasIndicators, color: '#8a94a6', fontSize: 10, hideOverlap: true }, axisLine: { lineStyle: { color: '#e9edf5' } } },
  ]
  const yAxes = [
    { scale: true, gridIndex: 0, position: 'left', axisLabel: { color: '#8a94a6' }, splitLine: { lineStyle: { color: '#e9edf5' } } },
    { scale: true, gridIndex: 1, position: 'left', axisLabel: { color: '#8a94a6' }, splitLine: { show: false }, min: 0 },
  ]
  const series = [
    { name: 'K线', type: 'candlestick', data: values, xAxisIndex: 0, yAxisIndex: 0,
      itemStyle: { color: '#ff4d4f', color0: '#21c08b', borderColor: '#ff4d4f', borderColor0: '#21c08b' } },
    { name: 'MA5', type: 'line', data: ma(5), smooth: true, showSymbol: false, xAxisIndex: 0, yAxisIndex: 0, lineStyle: { color: '#ff9800', width: 1 } },
    { name: 'MA10', type: 'line', data: ma(10), smooth: true, showSymbol: false, xAxisIndex: 0, yAxisIndex: 0, lineStyle: { color: '#2196f3', width: 1 } },
    { name: 'MA20', type: 'line', data: ma(20), smooth: true, showSymbol: false, xAxisIndex: 0, yAxisIndex: 0, lineStyle: { color: '#9c27b0', width: 1 } },
    { name: 'MA60', type: 'line', data: ma(60), smooth: true, showSymbol: false, xAxisIndex: 0, yAxisIndex: 0, lineStyle: { color: '#607d8b', width: 1, type: 'dashed' } },
    { name: 'MA120', type: 'line', data: ma(120), smooth: true, showSymbol: false, xAxisIndex: 0, yAxisIndex: 0, lineStyle: { color: '#c62828', width: 1, type: 'dashed' } },
    { name: 'MA250', type: 'line', data: ma(250), smooth: true, showSymbol: false, xAxisIndex: 0, yAxisIndex: 0, lineStyle: { color: '#1565c0', width: 1, type: 'dashed' } },
    { name: '成交量', type: 'bar', data: volumes, xAxisIndex: 1, yAxisIndex: 1,
      itemStyle: { color: params => (params.dataIndex > 0 && closes[params.dataIndex] >= closes[params.dataIndex-1] ? '#ff4d4f' : '#21c08b') } },
  ]
  const legendData = ['K线', 'MA5', 'MA10', 'MA20', 'MA60', 'MA120', 'MA250', '成交量']
  const indLegendData = []

  // 单选指标子图（仅一个，互斥显示）
  if (indicator === 'MACD') {
    grids.push({ left: 60, right: 20, top: 398, bottom: 64 })
    xAxes.push({ type: 'category', data: dates, gridIndex: 2, axisLabel: { color: '#8a94a6', fontSize: 10, hideOverlap: true }, axisTick: { show: false } })
    yAxes.push({ scale: true, gridIndex: 2, position: 'left', axisLabel: { color: '#8a94a6', fontSize: 10 }, splitLine: { lineStyle: { color: '#e9edf5' } } })
    const { dif, dea, bar } = calcMACD(closes)
    indLegendData.push('DIF', 'DEA', 'MACD')
    series.push(
      { name: 'DIF', type: 'line', data: dif, smooth: true, showSymbol: false, xAxisIndex: 2, yAxisIndex: 2, lineStyle: { color: '#00bcd4', width: 1 } },
      { name: 'DEA', type: 'line', data: dea, smooth: true, showSymbol: false, xAxisIndex: 2, yAxisIndex: 2, lineStyle: { color: '#ff9800', width: 1 } },
      { name: 'MACD', type: 'bar', data: bar, xAxisIndex: 2, yAxisIndex: 2, itemStyle: { color: p => bar[p.dataIndex] >= 0 ? '#ff4d4f' : '#21c08b' } },
    )
  } else if (indicator === 'KDJ') {
    grids.push({ left: 60, right: 20, top: 398, bottom: 64 })
    xAxes.push({ type: 'category', data: dates, gridIndex: 2, axisLabel: { color: '#8a94a6', fontSize: 10, hideOverlap: true }, axisTick: { show: false } })
    yAxes.push({ scale: true, gridIndex: 2, position: 'left', axisLabel: { color: '#8a94a6', fontSize: 10 }, splitLine: { lineStyle: { color: '#e9edf5' } } })
    const { k, d, j } = calcKDJ(highs, lows, closes)
    indLegendData.push('K', 'D', 'J')
    series.push(
      { name: 'K', type: 'line', data: k, smooth: true, showSymbol: false, xAxisIndex: 2, yAxisIndex: 2, lineStyle: { color: '#00bcd4', width: 1 } },
      { name: 'D', type: 'line', data: d, smooth: true, showSymbol: false, xAxisIndex: 2, yAxisIndex: 2, lineStyle: { color: '#ff9800', width: 1 } },
      { name: 'J', type: 'line', data: j, smooth: true, showSymbol: false, xAxisIndex: 2, yAxisIndex: 2, lineStyle: { color: '#e91e63', width: 1 } },
    )
  } else if (indicator === 'BOLL') {
    grids.push({ left: 60, right: 20, top: 398, bottom: 64 })
    xAxes.push({ type: 'category', data: dates, gridIndex: 2, axisLabel: { color: '#8a94a6', fontSize: 10, hideOverlap: true }, axisTick: { show: false } })
    yAxes.push({ scale: true, gridIndex: 2, position: 'left', axisLabel: { color: '#8a94a6', fontSize: 10 }, splitLine: { lineStyle: { color: '#e9edf5' } } })
    const { mid, upper, lower } = calcBOLL(closes)
    indLegendData.push('BOLL中轨', '上轨', '下轨')
    series.push(
      { name: 'BOLL中轨', type: 'line', data: mid, smooth: true, showSymbol: false, xAxisIndex: 2, yAxisIndex: 2, lineStyle: { color: '#ff9800', width: 1 } },
      { name: '上轨', type: 'line', data: upper, smooth: true, showSymbol: false, xAxisIndex: 2, yAxisIndex: 2, lineStyle: { color: '#2196f3', width: 1, type: 'dashed' } },
      { name: '下轨', type: 'line', data: lower, smooth: true, showSymbol: false, xAxisIndex: 2, yAxisIndex: 2, lineStyle: { color: '#2196f3', width: 1, type: 'dashed' }, areaStyle: { color: 'rgba(33,150,243,0.05)' } },
    )
  }

  const dzXAxisIndices = hasIndicators ? [0, 1, 2] : [0, 1]

  return {
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' }, confine: true },
    legend: [
      { data: legendData, top: 0, textStyle: { fontSize: 11, color: '#8a94a6' } },
      ...(indLegendData.length ? [{ data: indLegendData, top: 374, left: 60, itemGap: 8, itemWidth: 14, itemHeight: 8, textStyle: { fontSize: 10, color: '#8a94a6' } }] : []),
    ],
    grid: grids,
    xAxis: xAxes,
    yAxis: yAxes,
    dataZoom: [
      { type: 'inside', xAxisIndex: dzXAxisIndices, start: 60, end: 100 },
      { type: 'slider', xAxisIndex: dzXAxisIndices, start: 60, end: 100, height: 20, bottom: 8 },
    ],
    series,
  }
})

const timeshareOption = computed(() => {
  const data = timeshareData.value
  if (!data.length) return {}
  const times = data.map(d => d.time)
  const prices = data.map(d => d.price)
  const avgPrices = data.map(d => d.avgPrice)
  const vols = data.map(d => d.volume)
  const avgColor = '#f0a040'

  return {
    grid: { left: 70, right: 70, top: 20, bottom: 60 },
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    xAxis: {
      type: 'category', data: times, axisLabel: { interval: 30, rotate: 0, color: '#8e9bbd', fontSize: 10 },
      axisLine: { lineStyle: { color: '#23304f' } }, axisTick: { show: false },
    },
    yAxis: [
      { type: 'value', scale: true, splitLine: { lineStyle: { color: '#1e2845' } },
        axisLabel: { color: '#8e9bbd', fontSize: 10 }, position: 'right' },
      { type: 'value', scale: true, axisLabel: { show: false }, splitLine: { show: false } },
    ],
    dataZoom: [{ type: 'inside', start: 0, end: 100 }],
    series: [
      {
        name: '价格', type: 'line', data: prices, smooth: false, symbol: 'none',
        lineStyle: { color: '#4f8cff', width: 1.5 },
        areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [{offset:0,color:'rgba(79,140,255,0.15)'},{offset:1,color:'rgba(79,140,255,0.01)'}] } },
      },
      {
        name: '均价', type: 'line', data: avgPrices, smooth: false, symbol: 'none',
        lineStyle: { color: avgColor, width: 1, type: 'dashed' },
      },
      {
        name: '成交量', type: 'bar', data: vols, yAxisIndex: 1,
        itemStyle: { color: (p) => {
          const idx = p.dataIndex
          return idx > 0 && prices[idx] >= prices[idx-1] ? '#cc3d3d' : '#3dcc3d'
        } },
        barWidth: '100%',
      },
    ],
  }
})

async function refresh() {
  if (refreshing) return
  refreshing = true
  try {
    await store.refreshQuotes()
    if (klineFreq.value !== '分时') await loadKline(); else await loadTimeshare()
  } finally { refreshing = false }
}

watch(() => store.activeCode, (code) => {
  if (code) {
    if (klineFreq.value !== '分时') loadKline(); else loadTimeshare()
  }
})
watch(() => store.source, () => store.refreshQuotes().catch(() => {}))

onMounted(async () => {
  try { await store.fetchWatchlist() } catch (e) { toast('自选列表加载失败: ' + e.message) }
  await refresh().catch(() => {})
  timer = setInterval(() => { if (!document.hidden) store.refreshQuotes().catch(() => {}) }, 6000)
})
onUnmounted(() => clearInterval(timer))
</script>

<template>
  <div>
    <div class="toolbar-row">
      <select v-model="store.source" @change="store.setSource(store.source)">
        <option value="tencent">腾讯行情（推荐）</option>
        <option value="sina">新浪财经</option>
        <option value="eastmoney">东方财富</option>
      </select>
      <span class="hint" style="flex:0">{{ SOURCE_META[store.source] }}</span>
      <button class="btn-ghost" style="margin-left:auto" @click="refresh">手动刷新</button>
    </div>

    <div class="search-bar">
      <input v-model="codeInput" placeholder="输入股票代码添加自选，如 600519 / 000001 / 300750 / 830799 / bj920799，回车确认" @keydown.enter="addCode" />
      <button class="btn-primary" @click="addCode">添加</button>
    </div>

    <div class="detail-card">
      <template v-if="activeQuote">
        <div class="detail-head">
          <span class="detail-name">{{ activeQuote.name }}</span>
          <span class="detail-code">{{ store.activeCode }}</span>
          <span class="detail-time">{{ activeQuote.date }} {{ activeQuote.time }}</span>
        </div>
        <div class="price-block">
          <span class="price-main" :class="trendCls(activeQuote.price, activeQuote.preClose)">{{ fmtNum(activeQuote.price) }}</span>
          <div class="price-change" :class="trendCls(activeQuote.price, activeQuote.preClose)">
            <span>{{ activeQuote.price - activeQuote.preClose >= 0 ? '+' : '' }}{{ fmtNum(activeQuote.price - activeQuote.preClose) }}</span>
            <span>{{ fmtPct((activeQuote.price - activeQuote.preClose) / activeQuote.preClose * 100) }}</span>
          </div>
        </div>
        <div class="grid-4">
          <div class="item"><div class="label">今开</div><div class="value">{{ fmtNum(activeQuote.open) }}</div></div>
          <div class="item"><div class="label">昨收</div><div class="value">{{ fmtNum(activeQuote.preClose) }}</div></div>
          <div class="item"><div class="label">最高</div><div class="value up">{{ fmtNum(activeQuote.high) }}</div></div>
          <div class="item"><div class="label">最低</div><div class="value down">{{ fmtNum(activeQuote.low) }}</div></div>
          <div class="item"><div class="label">买一价</div><div class="value">{{ activeQuote.bid ? fmtNum(activeQuote.bid) : '--' }}</div></div>
          <div class="item"><div class="label">卖一价</div><div class="value">{{ activeQuote.ask ? fmtNum(activeQuote.ask) : '--' }}</div></div>
          <div class="item"><div class="label">成交量(手)</div><div class="value">{{ fmtVol(activeQuote.volume) }}</div></div>
          <div class="item"><div class="label">成交额(万)</div><div class="value">{{ fmtAmount(activeQuote.amount) }}</div></div>
        </div>
      </template>
      <div v-else class="empty-hint">从自选股列表选择一只查看详情，或在上方输入代码添加</div>
    </div>

    <div class="card chart-card mb-24">
      <div class="chart-toolbar">
        <div class="field" style="margin-right:12px">
          <select v-model="klineFreq" @change="klineFreq === '分时' ? loadTimeshare() : loadKline()" style="width:80px">
            <option value="分时">分时</option>
            <option value="D">日K</option>
            <option value="W">周K</option>
            <option value="M">月K</option>
          </select>
        </div>
        <div class="field">
          <label>天数</label>
          <input v-model.number="klineDays" type="number" min="30" max="1023" style="width:80px" @change="loadKline" />
        </div>
        <span class="sep">|</span>
        <button v-for="ind in ['MACD','KDJ','BOLL']" :key="ind"
          class="btn-ghost sm" :class="{ 'btn-active': activeIndicator === ind }"
          @click="toggleIndicator(ind)">{{ ind }}</button>
      </div>
      <template v-if="klineFreq === '分时'">
        <EChart v-if="timeshareData.length" :option="timeshareOption" height="520px" />
        <div v-else class="empty-hint">{{ timeshareWarning || '暂无分时数据' }}</div>
      </template>
      <template v-else>
        <EChart v-if="klineData.length" :option="klineOption" :height="chartHeight + 'px'" />
        <div v-else class="empty-hint">暂无K线数据</div>
      </template>
    </div>

    <div v-if="compareCodes.length" class="card mb-24">
      <div class="card-head">
        <h3>多股对比 · 基准收益曲线</h3>
        <button class="btn-ghost sm" @click="clearCompare">清除对比</button>
      </div>
      <EChart v-if="Object.keys(compareOption).length" :option="compareOption" height="360px" />
    </div>

    <div class="card">
      <div class="card-head"><h2>自选股</h2><span class="count">{{ store.codes.length ? `共 ${store.codes.length} 只` : '' }}</span></div>
      <table>
        <thead><tr><th>对比</th><th>名称</th><th>最新价</th><th>涨跌额</th><th>涨跌幅</th><th>最高</th><th>最低</th><th>成交量(手)</th><th>操作</th></tr></thead>
        <tbody>
          <tr v-if="!store.codes.length" class="empty-row"><td colspan="9">暂无自选股，在上方输入股票代码添加</td></tr>
          <tr v-for="code in store.codes" :key="code" class="clickable-row" :class="{ 'active-row': code === store.activeCode }" @click="selectCode(code)">
            <template v-if="store.quotes[code]">
              <td @click.stop><input type="checkbox" :checked="compareCodes.includes(code)" @change="toggleCompare(code)" /></td>
              <td class="td-name">{{ store.quotes[code].name }}<span class="td-code">{{ stripPrefix(code) }}</span></td>
              <td :class="trendCls(store.quotes[code].price, store.quotes[code].preClose)">{{ fmtNum(store.quotes[code].price) }}</td>
              <td :class="trendCls(store.quotes[code].price, store.quotes[code].preClose)">{{ (store.quotes[code].price - store.quotes[code].preClose >= 0 ? '+' : '') + fmtNum(store.quotes[code].price - store.quotes[code].preClose) }}</td>
              <td :class="trendCls(store.quotes[code].price, store.quotes[code].preClose)">{{ fmtPct((store.quotes[code].price - store.quotes[code].preClose) / store.quotes[code].preClose * 100) }}</td>
              <td>{{ fmtNum(store.quotes[code].high) }}</td>
              <td>{{ fmtNum(store.quotes[code].low) }}</td>
              <td>{{ fmtVol(store.quotes[code].volume) }}</td>
              <td><span class="del-btn" @click.stop="removeCode(code)">删除</span></td>
            </template>
            <template v-else>
              <td colspan="8" class="empty-row">{{ store.noData.includes(code) ? `${code} 暂无行情数据` : `${code} 加载中…` }}</td>
              <td><span class="del-btn" @click.stop="removeCode(code)">删除</span></td>
            </template>
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

.detail-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px 24px; margin-bottom: 16px; }
.detail-head { display: flex; align-items: baseline; gap: 12px; }
.detail-name { font-size: 18px; font-weight: 700; color: var(--text); }
.detail-code { font-size: 13px; color: var(--text-mute); font-family: monospace; }
.detail-time { margin-left: auto; font-size: 12px; color: var(--text-mute); }
.price-block { display: flex; align-items: baseline; gap: 14px; margin: 10px 0; }
.price-main { font-size: 28px; font-weight: 700; }
.price-change { display: flex; gap: 8px; font-size: 14px; }
.grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 14px; }
.grid-4 .item { background: var(--bg-2); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; text-align: center; }
.grid-4 .label { font-size: 11px; color: var(--text-mute); }
.grid-4 .value { font-size: 15px; font-weight: 600; color: var(--text); margin-top: 4px; }

.chart-card { position: relative; }
.chart-toolbar { display: flex; align-items: center; gap: 6px; margin-bottom: 10px; padding: 6px 0; }
.chart-toolbar .sep { color: var(--border); font-size: 16px; margin: 0 6px; }
.btn-active { background: var(--accent) !important; color: #fff !important; border-color: var(--accent) !important; }

table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { background: var(--card-2); color: var(--text-dim); font-weight: 600; padding: 10px 12px; text-align: left; border-bottom: 2px solid var(--border); position: sticky; top: 0; }
td { padding: 8px 12px; border-bottom: 1px solid var(--border); color: var(--text); }
.clickable-row { cursor: pointer; }
.clickable-row:hover { background: var(--bg-2); }
.active-row { background: rgba(79,140,255,.08); }
.td-name { font-weight: 600; }
.td-code { display: block; font-size: 11px; color: var(--text-mute); font-family: monospace; }
.empty-row { text-align: center; color: var(--text-mute); }
.empty-hint { text-align: center; color: var(--text-mute); padding: 30px; font-size: 14px; }
.del-btn { color: #ff6b6b; cursor: pointer; font-size: 12px; }
.del-btn:hover { text-decoration: underline; }

.up { color: #ff4d4f; }
.down { color: #21c08b; }

.toolbar-row { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
.toolbar-row select { width: 160px; }
.search-bar { display: flex; gap: 10px; margin-bottom: 16px; }
.search-bar input { flex: 1; }
.field { display: flex; align-items: center; gap: 6px; }
.field label { font-size: 12px; color: var(--text-mute); white-space: nowrap; }
.card-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }
.card-head h2, .card-head h3 { color: var(--text); }
.count { font-size: 12px; color: var(--text-mute); }
.mb-24 { margin-bottom: 24px; }
</style>
