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
let timer = null
let refreshing = false

const SOURCE_META = {
  tencent: '腾讯行情（推荐） · qt.gtimg.cn',
  sina: '新浪财经 · 可能受防盗链限制',
  eastmoney: '东方财富 · JSON'
}

const activeQuote = computed(() => store.quotes[store.activeCode])

// ---- 技术指标 ----
const activeIndicators = ref([])
const INDICATOR_COLORS = ['#00bcd4', '#ff9800', '#e91e63']

const chartHeight = computed(() => {
  const n = activeIndicators.value.length
  return n === 0 ? 520 : n === 1 ? 620 : n === 2 ? 720 : 820
})

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
  const idx = activeIndicators.value.indexOf(name)
  if (idx >= 0) activeIndicators.value.splice(idx, 1)
  else activeIndicators.value.push(name)
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
  }
})

async function addCode() {
  const c = normalizeCode(codeInput.value)
  if (!c) { toast('代码格式错误，应为 6 位数字或 sh/sz/bj 前缀'); return }
  if (store.codes.includes(c)) { toast('已在自选列表中'); return }
  try {
    // 先校验代码是否存在：不存在直接提示，不入自选（旧版会加进去一直"加载中"）
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

  const indicators = activeIndicators.value
  const hasIndicators = indicators.length > 0

  // 动态构建 grid
  const grids = [
    { left: 60, right: 20, top: 40, bottom: hasIndicators ? '60%' : '52%' },
    { left: 60, right: 20, top: hasIndicators ? '45%' : '58%', bottom: hasIndicators ? '32%' : 60 },
  ]
  const xAxes = [
    { type: 'category', data: dates, gridIndex: 0, axisLabel: { show: false }, axisTick: { show: false } },
    { type: 'category', data: dates, gridIndex: 1, axisLabel: { color: '#8a94a6' }, axisLine: { lineStyle: { color: '#e9edf5' } } },
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

  // 动态追加强标子图
  const dzXAxisIndexBase = 2
  let indicatorGridTop = 0.72
  indicators.forEach((name, idx) => {
    const gi = 2 + idx
    grids.push({ left: 60, right: 20, top: `${indicatorGridTop * 100}%`, bottom: idx === indicators.length - 1 ? 60 : `${(indicatorGridTop + 0.14) * 100 + 1}%` })
    xAxes.push({ type: 'category', data: dates, gridIndex: gi, axisLabel: { show: idx === indicators.length - 1, color: '#8a94a6', rotate: 30 }, axisTick: { show: false } })
    yAxes.push({ scale: true, gridIndex: gi, position: 'left', axisLabel: { color: '#8a94a6', fontSize: 10 }, splitLine: { lineStyle: { color: '#e9edf5' } } })
    indicatorGridTop -= 0.14

    if (name === 'MACD') {
      const { dif, dea, bar } = calcMACD(closes)
      legendData.push('DIF', 'DEA', 'MACD')
      series.push(
        { name: 'DIF', type: 'line', data: dif, smooth: true, showSymbol: false, xAxisIndex: gi, yAxisIndex: gi, lineStyle: { color: '#00bcd4', width: 1 } },
        { name: 'DEA', type: 'line', data: dea, smooth: true, showSymbol: false, xAxisIndex: gi, yAxisIndex: gi, lineStyle: { color: '#ff9800', width: 1 } },
        { name: 'MACD', type: 'bar', data: bar, xAxisIndex: gi, yAxisIndex: gi,
          itemStyle: { color: p => bar[p.dataIndex] >= 0 ? '#ff4d4f' : '#21c08b' } },
      )
    } else if (name === 'KDJ') {
      const { k, d, j } = calcKDJ(highs, lows, closes)
      legendData.push('K', 'D', 'J')
      series.push(
        { name: 'K', type: 'line', data: k, smooth: true, showSymbol: false, xAxisIndex: gi, yAxisIndex: gi, lineStyle: { color: '#00bcd4', width: 1 } },
        { name: 'D', type: 'line', data: d, smooth: true, showSymbol: false, xAxisIndex: gi, yAxisIndex: gi, lineStyle: { color: '#ff9800', width: 1 } },
        { name: 'J', type: 'line', data: j, smooth: true, showSymbol: false, xAxisIndex: gi, yAxisIndex: gi, lineStyle: { color: '#e91e63', width: 1 } },
      )
    } else if (name === 'BOLL') {
      const { mid, upper, lower } = calcBOLL(closes)
      legendData.push('BOLL中轨', '上轨', '下轨')
      series.push(
        { name: 'BOLL中轨', type: 'line', data: mid, smooth: true, showSymbol: false, xAxisIndex: gi, yAxisIndex: gi, lineStyle: { color: '#ff9800', width: 1 } },
        { name: '上轨', type: 'line', data: upper, smooth: true, showSymbol: false, xAxisIndex: gi, yAxisIndex: gi, lineStyle: { color: '#2196f3', width: 1, type: 'dashed' } },
        { name: '下轨', type: 'line', data: lower, smooth: true, showSymbol: false, xAxisIndex: gi, yAxisIndex: gi, lineStyle: { color: '#2196f3', width: 1, type: 'dashed' }, areaStyle: { color: 'rgba(33,150,243,0.05)' } },
      )
    }
  })

  const dzXAxisIndices = hasIndicators ? Array.from({ length: 2 + indicators.length }, (_, i) => i) : [0, 1]

  return {
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    legend: { data: legendData, top: 0, textStyle: { fontSize: 11, color: '#8a94a6' } },
    grid: grids,
    xAxis: xAxes,
    yAxis: yAxes,
    dataZoom: [
      { type: 'inside', xAxisIndex: dzXAxisIndices, start: 60, end: 100 },
      { type: 'slider', xAxisIndex: dzXAxisIndices, start: 60, end: 100, height: 24, bottom: 10 },
    ],
    series,
  }
})

async function refresh() {
  if (refreshing) return   // 在途锁：上次未完成则跳过
  refreshing = true
  try {
    await store.refreshQuotes()
    await loadKline()
  } finally { refreshing = false }
}

watch(() => store.activeCode, loadKline)
watch(() => store.source, () => store.refreshQuotes().catch(() => {}))

onMounted(async () => {
  // 任一接口失败不阻塞页面初始化与定时刷新（旧版 await 抛错会跳过 timer，页面停摆）
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
          <select v-model="klineFreq" @change="loadKline" style="width:80px">
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
          class="btn-ghost sm" :class="{ 'btn-active': activeIndicators.includes(ind) }"
          @click="toggleIndicator(ind)">{{ ind }}</button>
      </div>
      <EChart v-if="klineData.length" :option="klineOption" :height="chartHeight + 'px'" />
      <div v-else class="empty-hint">暂无K线数据</div>
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
