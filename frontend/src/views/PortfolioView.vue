<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { usePortfolioStore } from '../stores/portfolio'
import { useWatchlistStore } from '../stores/watchlist'
import { useToast } from '../stores/toast'
import EChart from '../components/EChart.vue'
import { fmtNum, fmtMoney, fmtPct, stripPrefix } from '../utils/format'

const portfolio = usePortfolioStore()
const watchlist = useWatchlistStore()
const { toast } = useToast()

const orderCode = ref('')
const orderSide = ref('buy')
const orderQty = ref(100)
let timer = null

const orderPrice = computed(() => watchlist.quotes[orderCode.value]?.price || 0)
const orderEstimate = computed(() => orderPrice.value ? orderPrice.value * orderQty.value : null)

async function placeOrder() {
  if (!orderCode.value) { toast('请先添加自选股'); return }
  if (!orderQty.value || orderQty.value % 100 !== 0) { toast('数量需为100的整数倍'); return }
  try {
    await portfolio.order(orderCode.value, orderSide.value, Number(orderQty.value))
    toast(orderSide.value === 'buy' ? '买入成功' : '卖出成功')
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
    xAxis: { type: 'category', data: labels, axisLine: { lineStyle: { color: '#2a3354' } }, axisLabel: { color: '#7c89a8' } },
    yAxis: { type: 'value', axisLine: { lineStyle: { color: '#2a3354' } }, axisLabel: { color: '#7c89a8' }, splitLine: { lineStyle: { color: '#1c2238' } } },
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
  if (watchlist.codes.length) orderCode.value = watchlist.codes[0]
  timer = setInterval(async () => {
    await watchlist.refreshQuotes()
    await portfolio.fetch()
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
      <select v-model="orderCode">
        <option v-if="!watchlist.codes.length" value="">请先添加自选股</option>
        <option v-for="code in watchlist.codes" :key="code" :value="code">
          {{ watchlist.quotes[code]?.name || code }}（{{ stripPrefix(code) }}）
        </option>
      </select>
      <div class="side-toggle">
        <button class="side-btn" :class="{ active: orderSide === 'buy', buy: true }" @click="orderSide = 'buy'">买入</button>
        <button class="side-btn" :class="{ active: orderSide === 'sell', sell: true }" @click="orderSide = 'sell'">卖出</button>
      </div>
      <input v-model="orderQty" type="number" step="100" min="100" />
      <span class="order-estimate">预计金额：{{ orderEstimate ? fmtMoney(orderEstimate) : '--' }}</span>
      <button class="btn-primary" @click="placeOrder">下单</button>
      <button class="btn-ghost" @click="resetPortfolio">重置模拟盘</button>
    </div>

    <div class="card mb-24">
      <div class="card-head"><h2>持仓</h2><span class="count">{{ portfolio.positions.length ? `共 ${portfolio.positions.length} 只` : '' }}</span></div>
      <table>
        <thead><tr><th>名称</th><th>数量</th><th>成本价</th><th>现价</th><th>市值</th><th>浮动盈亏</th><th>盈亏率</th></tr></thead>
        <tbody>
          <tr v-if="!portfolio.positions.length" class="empty-row"><td colspan="7">暂无持仓</td></tr>
          <tr v-for="p in portfolio.positions" :key="p.code">
            <td class="td-name">{{ p.name }}<span class="td-code">{{ stripPrefix(p.code) }}</span></td>
            <td>{{ p.qty }}</td>
            <td>{{ fmtNum(p.avgCost) }}</td>
            <td>{{ fmtNum(p.price) }}</td>
            <td>{{ p.marketValue.toLocaleString(undefined, { maximumFractionDigits: 2 }) }}</td>
            <td :class="p.pnl > 0 ? 'up' : p.pnl < 0 ? 'down' : 'flat'">{{ p.pnl >= 0 ? '+' : '' }}{{ p.pnl.toFixed(2) }}</td>
            <td :class="p.pnl > 0 ? 'up' : p.pnl < 0 ? 'down' : 'flat'">{{ fmtPct(p.pnlPct) }}</td>
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
