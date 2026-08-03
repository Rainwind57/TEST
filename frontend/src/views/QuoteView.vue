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
let timer = null
let refreshing = false   // 在途锁：弱网下上次刷新未完成则跳过，避免请求堆积重叠

const SOURCE_META = {
  tencent: '腾讯行情（推荐） · qt.gtimg.cn',
  sina: '新浪财经 · 可能受防盗链限制',
  eastmoney: '东方财富 · JSON'
}

const activeQuote = computed(() => store.quotes[store.activeCode])

async function addCode() {
  const c = normalizeCode(codeInput.value)
  if (!c) { toast('代码格式错误，应为 6 位数字或 sh/sz 前缀'); return }
  if (store.codes.includes(c)) { toast('已在自选列表中'); return }
  try {
    await store.addCode(c)
    codeInput.value = ''
    toast('已添加 ' + c)
  } catch (e) { toast(e.message) }
}

async function removeCode(code) {
  try { await store.removeCode(code); toast('已删除') } catch (e) { toast(e.message) }
}

function selectCode(code) { store.activeCode = code }

async function loadKline() {
  if (!store.activeCode) { klineData.value = []; return }
  try {
    const res = await api.get('/kline', { params: { code: store.activeCode, days: 90 } })
    klineData.value = res.data
  } catch (e) { klineData.value = [] }
}

const klineOption = computed(() => {
  const dates = klineData.value.map(k => k.date)
  const values = klineData.value.map(k => [k.open, k.close, k.low, k.high])
  return {
    backgroundColor: 'transparent',
    grid: { left: 50, right: 20, top: 20, bottom: 30 },
    xAxis: { type: 'category', data: dates, axisLine: { lineStyle: { color: '#d7dce8' } }, axisLabel: { color: '#8a94a6' } },
    yAxis: { scale: true, axisLine: { lineStyle: { color: '#d7dce8' } }, axisLabel: { color: '#8a94a6' }, splitLine: { lineStyle: { color: '#e9edf5' } } },
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    series: [{
      type: 'candlestick', data: values,
      itemStyle: { color: '#ff4d4f', color0: '#21c08b', borderColor: '#ff4d4f', borderColor0: '#21c08b' }
    }]
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
watch(() => store.source, () => store.refreshQuotes())

onMounted(async () => {
  await store.fetchWatchlist()
  await refresh()
  timer = setInterval(() => { if (!document.hidden) store.refreshQuotes() }, 6000)
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
      <input v-model="codeInput" placeholder="输入股票代码添加自选，如 600519 / 000001 / 300750，回车确认" @keydown.enter="addCode" />
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
      <EChart v-if="klineData.length" :option="klineOption" height="320px" />
      <div v-else class="empty-hint">暂无K线数据</div>
    </div>

    <div class="card">
      <div class="card-head"><h2>自选股</h2><span class="count">{{ store.codes.length ? `共 ${store.codes.length} 只` : '' }}</span></div>
      <table>
        <thead><tr><th>名称</th><th>最新价</th><th>涨跌额</th><th>涨跌幅</th><th>最高</th><th>最低</th><th>成交量(手)</th><th>操作</th></tr></thead>
        <tbody>
          <tr v-if="!store.codes.length" class="empty-row"><td colspan="8">暂无自选股，在上方输入股票代码添加</td></tr>
          <tr v-for="code in store.codes" :key="code" class="clickable-row" :class="{ 'active-row': code === store.activeCode }" @click="selectCode(code)">
            <template v-if="store.quotes[code]">
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
              <td colspan="8" class="empty-row">{{ code }} 加载中…</td>
            </template>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
