<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { usePortfolioStore } from '../stores/portfolio'
import { useWatchlistStore } from '../stores/watchlist'
import { useToast } from '../stores/toast'
import EChart from '../components/EChart.vue'
import api from '../api/client'
import { fmtNum, fmtMoney, fmtPct, stripPrefix, normalizeCode } from '../utils/format'

const portfolio = usePortfolioStore()
const watchlist = useWatchlistStore()
const { toast } = useToast()

const orderCode = ref('')
const orderSide = ref('buy')
const orderQty = ref(100)
const orderQuote = ref(null)      // 下单标的独立行情（解耦自选股）
const orderValidate = ref(null)   // { code, name, tradable }
const boardOptions = ref([])      // 板块列表（P1 板块选股入口）
const boardStocks = ref([])       // 所选板块的股票列表（供联想下拉）
let timer = null
let polling = false   // 在途锁：弱网下上次轮询未完成则跳过，避免请求堆积

const orderPrice = computed(() => orderQuote.value?.price || 0)
const orderEstimate = computed(() => orderPrice.value ? orderPrice.value * orderQty.value : null)

// 代码输入：归一化 + 防抖拉取单票行情（非自选股也能看到现价）
let debounceTimer = null
async function onCodeInput() {
  clearTimeout(debounceTimer)
  debounceTimer = setTimeout(async () => {
    const code = normalizeCode(orderCode.value)
    if (!code) { orderQuote.value = null; orderValidate.value = null; return }
    try {
      const [quoteData, validateData] = await Promise.all([
        api.get('/quote', { params: { codes: code } }),
        api.get('/quote/validate', { params: { code } }),
      ])
      orderQuote.value = quoteData[code] || null
      orderValidate.value = validateData
    } catch (e) {
      orderQuote.value = null
      orderValidate.value = null
    }
  }, 300)
}

function onCodeBlur() {
  // 失焦时归一化输入（支持纯数字 → sh/sz/bj 前缀）
  const code = normalizeCode(orderCode.value)
  if (code) orderCode.value = code
}

// P1：板块选股入口（下拉选板块 → 加载该板块股票到联想列表）
const selectedBoard = ref('all')
async function loadBoards() {
  try {
    const boards = await api.get('/select/boards')
    boardOptions.value = [{ value: 'all', label: '全部A股' }, ...(boards || [])]
  } catch (e) { /* 静默 */ }
}
async function onBoardChange() {
  boardStocks.value = []
  try {
    const rows = await api.get('/select/market', { params: { board: selectedBoard.value, limit: 200 } })
    boardStocks.value = (rows || []).map(r => ({ code: r.code, name: r.name || '' }))
  } catch (e) { toast(e.message) }
}
function pickBoardStock(code) {
  orderCode.value = code
  onCodeInput()
}

// 持仓行内买卖：预填下单面板（代码 / 方向 / 数量），复用现有 order 流程
function prefillOrder(p, side) {
  orderCode.value = p.code
  orderSide.value = side
  orderQty.value = p.qty
  onCodeInput()
}

// P1：下单成功后提示加入自选（便捷而非门槛）
function addCurrentToWatchlist() {
  const code = normalizeCode(orderCode.value)
  if (!code) { toast('请先输入有效股票代码'); return }
  if (watchlist.codes.includes(code)) { toast('已在自选中'); return }
  watchlist.addCode(code, orderQuote.value?.name || orderValidate.value?.name || '')
    .then(() => toast('已加入自选'))
    .catch(e => toast(e.message))
}

async function placeOrder() {
  const code = normalizeCode(orderCode.value)
  if (!code) { toast('请输入股票代码（如 sh600519 / bj830799）'); return }
  if (orderValidate.value && !orderValidate.value.tradable) {
    toast('该代码为指数/ETF，不可交易'); return
  }
  if (!orderQuote.value?.price) { toast('暂无行情，请确认代码后重试'); return }
  if (!orderQty.value || orderQty.value % 100 !== 0) { toast('数量需为100的整数倍'); return }
  try {
    await portfolio.order(code, orderSide.value, Number(orderQty.value))
    const sideLabel = { buy: '买入', sell: '卖出', short: '做空', cover: '回补' }
    toast(`${sideLabel[orderSide.value] || '下单'}成功`)
  } catch (e) {
    toast(e.message)
  }
}

async function resetPortfolio() {
  if (!confirm('确认重置模拟盘？现金将恢复为初始值，持仓和成交记录会清空。')) return
  await portfolio.reset()
  toast('模拟盘已重置')
}

const equityOption = computed(() => {
  const labels = portfolio.equity.map(e => new Date(e.ts).toLocaleString().slice(5, 16))
  const values = portfolio.equity.map(e => e.value)
  return {
    backgroundColor: 'transparent',
    grid: { left: 60, right: 20, top: 20, bottom: 40 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: labels, axisLine: { lineStyle: { color: '#d7dce8' } }, axisLabel: { color: '#8a94a6' } },
    yAxis: { type: 'value', axisLine: { lineStyle: { color: '#d7dce8' } }, axisLabel: { color: '#8a94a6' }, splitLine: { lineStyle: { color: '#e9edf5' } } },
    series: [{
      type: 'line', data: values, showSymbol: false, smooth: true,
      lineStyle: { color: '#4f8cff', width: 2 },
      areaStyle: { color: 'rgba(79,140,255,.12)' }
    }]
  }
})

onMounted(async () => {
  if (!watchlist.codes.length) await watchlist.fetchWatchlist()
  await watchlist.refreshQuotes()
  await portfolio.fetch()
  loadBoards()
  if (watchlist.codes.length) {
    orderCode.value = watchlist.codes[0]
    await onCodeInput()
  }
  timer = setInterval(async () => {
    if (document.hidden || polling) return   // 在途锁
    polling = true
    try {
      await watchlist.refreshQuotes()
      await portfolio.fetch()
      // 下单标的非自选股时，也按需刷新其报价
      const oc = normalizeCode(orderCode.value)
      if (oc && !watchlist.codes.includes(oc)) {
        const q = await api.get('/quote', { params: { codes: oc } })
        orderQuote.value = q[oc] || orderQuote.value
      }
    } finally { polling = false }
  }, 6000)
})
onUnmounted(() => clearInterval(timer))
</script>

<template>
  <div>
    <div class="asset-cards">
      <div class="asset-card"><div class="label">总资产</div><div class="value">{{ fmtMoney(portfolio.totalAssets) }}</div></div>
      <div class="asset-card"><div class="label">可用现金</div><div class="value">{{ fmtMoney(portfolio.cash) }}</div></div>
      <div class="asset-card"><div class="label">持仓市值</div><div class="value">{{ fmtMoney(portfolio.marketValue) }}</div></div>
      <div class="asset-card">
        <div class="label">累计盈亏</div>
        <div class="value" :class="portfolio.totalPnl >= 0 ? 'up' : 'down'">
          {{ portfolio.totalPnl >= 0 ? '+' : '' }}{{ fmtMoney(portfolio.totalPnl) }} ({{ fmtPct(portfolio.totalPnlPct) }})
        </div>
      </div>
    </div>

    <div class="card chart-card mb-24">
      <EChart v-if="portfolio.equity.length" :option="equityOption" height="300px" />
    </div>

    <div class="order-panel">
      <div class="code-input-wrap">
        <input
          v-model="orderCode"
          @input="onCodeInput"
          @blur="onCodeBlur"
          list="watchlist-suggestions"
          placeholder="输入代码，如 sh600519 / bj830799"
          class="code-input"
        />
        <datalist id="watchlist-suggestions">
          <option v-for="code in watchlist.codes" :key="code" :value="code">
            {{ watchlist.quotes[code]?.name || code }}（{{ stripPrefix(code) }}）
          </option>
          <option v-for="s in boardStocks" :key="s.code" :value="s.code">{{ s.name }}（{{ stripPrefix(s.code) }}）</option>
        </datalist>
        <span v-if="orderQuote?.name" class="quote-name">{{ orderQuote.name }}</span>
        <span v-if="orderValidate && !orderValidate.tradable" class="quote-warn">指数/ETF 不可交易</span>
      </div>
      <select v-model="selectedBoard" @change="onBoardChange" class="board-select" title="从板块挑股直接下单，无需加入自选">
        <option v-for="b in boardOptions" :key="b.value" :value="b.value">{{ b.label }}</option>
      </select>
      <select v-if="boardStocks.length" @change="pickBoardStock($event.target.value)" class="stock-select" title="该板块股票列表">
        <option value="">选股…</option>
        <option v-for="s in boardStocks" :key="s.code" :value="s.code">{{ s.name }}（{{ stripPrefix(s.code) }}）</option>
      </select>
      <div class="side-toggle">
        <button class="side-btn" :class="{ active: orderSide === 'buy', buy: true }" @click="orderSide = 'buy'">买入</button>
        <button class="side-btn" :class="{ active: orderSide === 'sell', sell: true }" @click="orderSide = 'sell'">卖出</button>
      </div>
      <input v-model="orderQty" type="number" step="100" min="100" />
      <span class="order-estimate">现价 {{ orderPrice ? fmtNum(orderPrice) : '--' }} ｜ 预计金额：{{ orderEstimate ? fmtMoney(orderEstimate) : '--' }}</span>
      <button class="btn-primary" @click="placeOrder">下单</button>
      <button class="btn-ghost" @click="addCurrentToWatchlist" title="下单标的不必在自选中，可选加入以持续监控">加入自选</button>
      <button class="btn-ghost" @click="resetPortfolio">重置模拟盘</button>
    </div>

    <div class="card mb-24">
      <div class="card-head"><h2>持仓</h2><span class="count">{{ portfolio.positions.length ? `共 ${portfolio.positions.length} 只` : '' }}</span></div>
      <table>
        <thead><tr><th>名称</th><th>数量</th><th>成本价</th><th>现价</th><th>市值</th><th>浮动盈亏</th><th>盈亏率</th><th>操作</th></tr></thead>
        <tbody>
          <tr v-if="!portfolio.positions.length" class="empty-row"><td colspan="8">暂无持仓</td></tr>
          <tr v-for="p in portfolio.positions" :key="p.code">
            <td class="td-name">{{ p.name }}<span class="td-code">{{ stripPrefix(p.code) }}</span></td>
            <td>{{ p.qty }}</td>
            <td>{{ fmtNum(p.avgCost) }}</td>
            <td>{{ fmtNum(p.price) }}</td>
            <td>{{ p.marketValue.toLocaleString(undefined, { maximumFractionDigits: 2 }) }}</td>
            <td :class="p.pnl > 0 ? 'up' : p.pnl < 0 ? 'down' : 'flat'">{{ p.pnl >= 0 ? '+' : '' }}{{ p.pnl.toFixed(2) }}</td>
            <td :class="p.pnl > 0 ? 'up' : p.pnl < 0 ? 'down' : 'flat'">{{ fmtPct(p.pnlPct) }}</td>
            <td class="td-actions">
              <template v-if="p.side !== 'short'">
                <button class="act-buy" @click="prefillOrder(p, 'buy')" title="预填买入（加仓）">买入</button>
                <button class="act-sell" @click="prefillOrder(p, 'sell')" title="预填卖出（默认数量=持仓量，即清仓）">卖出</button>
              </template>
              <button v-else class="act-sell" @click="prefillOrder(p, 'cover')" title="回补空单（默认数量=持仓量，即平仓）">回补</button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="card">
      <div class="card-head"><h3>成交记录</h3><span class="count">{{ portfolio.trades.length ? `共 ${portfolio.trades.length} 笔` : '' }}</span></div>
      <table>
        <thead><tr><th>时间</th><th>方向</th><th>名称</th><th>数量</th><th>价格</th><th>金额</th></tr></thead>
        <tbody>
          <tr v-if="!portfolio.trades.length" class="empty-row"><td colspan="6">暂无成交</td></tr>
          <tr v-for="(t, idx) in portfolio.trades" :key="idx">
            <td style="font-family:inherit">{{ t.time }}</td>
            <td :class="t.side === 'buy' ? 'up' : 'down'">{{ t.side === 'buy' ? '买入' : '卖出' }}</td>
            <td class="td-name">{{ t.name }}</td>
            <td>{{ t.qty }}</td>
            <td>{{ fmtNum(t.price) }}</td>
            <td>{{ t.amount.toLocaleString(undefined, { maximumFractionDigits: 2 }) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
