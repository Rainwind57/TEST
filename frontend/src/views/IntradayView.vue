<script setup>
import { ref } from 'vue'
import api from '../api/client'
import { useToast } from '../stores/toast'
import EChart from '../components/EChart.vue'
import { computed } from 'vue'

const { toast } = useToast()

const code = ref('sh600519')
const period = ref('5')
const count = ref(240)
const signalLookback = ref(10)
const entryThreshold = ref(0.005)
const takeProfit = ref(0.02)
const stopLoss = ref(-0.01)
const sharesPerTrade = ref(100)
const maxTrades = ref(10)
// 成本参数（P3：旧版 saveStrategy 漏成本，回放时丢失口径）
const commissionRate = ref(0.00025)
const stampDuty = ref(0.001)
const slippage = ref(0.001)
const applyCost = ref(true)

const loading = ref(false)
const result = ref(null)

async function run() {
  loading.value = true
  try {
    result.value = await api.post('/intraday/backtest', {
      code: code.value, period: period.value, count: Number(count.value),
      signalLookback: Number(signalLookback.value), entryThreshold: Number(entryThreshold.value),
      takeProfit: Number(takeProfit.value), stopLoss: Number(stopLoss.value),
      sharesPerTrade: Number(sharesPerTrade.value), maxTrades: Number(maxTrades.value),
      commissionRate: Number(commissionRate.value), stampDuty: Number(stampDuty.value),
      slippage: Number(slippage.value), applyCost: applyCost.value,
    })
    toast(`回测完成，${result.value.metrics.nTrades} 笔交易`)
  } catch (e) { toast(e.message) }
  finally { loading.value = false }
}

async function saveStrategy() {
  const name = prompt('策略名称', `日内_${code.value}_${period.value}m`)
  if (!name) return
  try {
    await api.post('/strategies', {
      name, kind: 'intraday',
      config: {
        code: code.value, period: period.value, count: Number(count.value),
        signalLookback: Number(signalLookback.value), entryThreshold: Number(entryThreshold.value),
        takeProfit: Number(takeProfit.value), stopLoss: Number(stopLoss.value),
        sharesPerTrade: Number(sharesPerTrade.value), maxTrades: Number(maxTrades.value),
        commissionRate: Number(commissionRate.value), stampDuty: Number(stampDuty.value),
        slippage: Number(slippage.value), applyCost: applyCost.value,
      },
    })
    toast('策略已保存，可在策略中心一键运行')
  } catch (e) { toast(e.message) }
}

const fmt = v => v == null ? '-' : Number(v).toFixed(4)
const fmtPct = v => v == null ? '-' : (Number(v) * 100).toFixed(2) + '%'

const pnlOption = computed(() => {
  if (!result.value?.trades?.length) return {}
  let cum = 0
  const data = result.value.trades.map(t => {
    cum += t.pnl
    return [t.exit_time, cum]
  })
  return {
    tooltip: { trigger: 'axis' },
    grid: { left: 60, right: 20, top: 30, bottom: 40 },
    xAxis: { type: 'category', data: data.map(d => d[0]) },
    yAxis: { type: 'value', name: '累计盈亏' },
    series: [{ type: 'line', data: data.map(d => d[1]), smooth: true,
      itemStyle: { color: '#4f8cff' }, areaStyle: { opacity: 0.15 } }]
  }
})
</script>

<template>
  <div>
    <div class="card">
      <div class="card-head"><h3>分钟级回测</h3><span class="hint">日内动量信号 + 止盈止损</span></div>
      <div class="panel-toolbar">
        <div class="field"><label>股票代码</label><input v-model="code" /></div>
        <div class="field"><label>周期(分钟)</label>
          <select v-model="period">
            <option value="1">1</option><option value="5">5</option>
            <option value="15">15</option><option value="30">30</option><option value="60">60</option>
          </select>
        </div>
        <div class="field"><label>K线数</label><input v-model="count" type="number" /></div>
        <div class="field"><label>信号回看</label><input v-model="signalLookback" type="number" /></div>
        <div class="field"><label>进场阈值</label><input v-model="entryThreshold" type="number" step="0.001" /></div>
      </div>
      <div class="panel-toolbar" style="margin-top:10px">
        <div class="field"><label>止盈</label><input v-model="takeProfit" type="number" step="0.005" /></div>
        <div class="field"><label>止损</label><input v-model="stopLoss" type="number" step="0.005" /></div>
        <div class="field"><label>每笔股数</label><input v-model="sharesPerTrade" type="number" /></div>
        <div class="field"><label>最大交易数</label><input v-model="maxTrades" type="number" /></div>
      </div>
      <div class="panel-toolbar" style="margin-top:10px">
        <div class="field"><label>佣金率</label><input v-model="commissionRate" type="number" step="0.00005" :disabled="!applyCost" /></div>
        <div class="field"><label>印花税</label><input v-model="stampDuty" type="number" step="0.0005" :disabled="!applyCost" /></div>
        <div class="field"><label>滑点</label><input v-model="slippage" type="number" step="0.0005" :disabled="!applyCost" /></div>
        <div class="field"><label class="switch-label" style="font-weight:400"><input type="checkbox" v-model="applyCost" /> 计入成本</label></div>
      </div>
      <button class="btn-primary" style="margin-top:10px" :disabled="loading" @click="run">{{ loading ? '回测中…' : '开始回测' }}</button>
      <button class="btn-ghost" style="margin-top:10px;margin-left:8px" @click="saveStrategy">保存为策略</button>
    </div>

    <div v-if="result" class="card">
      <h3>回测结果 · {{ result.nBars }} 根 K 线</h3>
      <div class="kpi-row">
        <div class="kpi"><div class="n">{{ result.metrics.nTrades }}</div><div class="l">交易数</div></div>
        <div class="kpi"><div class="n">{{ fmtPct(result.metrics.winRate) }}</div><div class="l">胜率</div></div>
        <div class="kpi"><div class="n">{{ fmt(result.metrics.totalPnl) }}</div><div class="l">总盈亏</div></div>
        <div class="kpi"><div class="n">{{ fmt(result.metrics.sharpe) }}</div><div class="l">Sharpe</div></div>
      </div>
      <table class="data-table">
        <thead><tr><th>进场</th><th>进场价</th><th>出场</th><th>出场价</th><th>股数</th><th>盈亏</th><th>原因</th></tr></thead>
        <tbody>
          <tr v-for="(t, i) in result.trades" :key="i">
            <td>{{ t.entry_time }}</td><td>{{ fmt(t.entry_price) }}</td>
            <td>{{ t.exit_time }}</td><td>{{ fmt(t.exit_price) }}</td>
            <td>{{ t.shares }}</td>
            <td :class="t.pnl >= 0 ? 'up' : 'down'">{{ fmt(t.pnl) }}</td>
            <td><span class="tag">{{ t.reason }}</span></td>
          </tr>
        </tbody>
      </table>
    </div>

    <div v-if="result?.trades?.length" class="card">
      <h3>累计盈亏曲线</h3>
      <EChart :option="pnlOption" style="height:280px" />
    </div>
  </div>
</template>

<style scoped>
.data-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
.data-table th, .data-table td { border: 1px solid var(--border); padding: 8px 11px; text-align: left; }
.data-table th { background: var(--card-2); color: var(--text-dim); font-weight: 600; }
.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px; margin: 14px 0; }
.kpi { background: var(--card-2); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }
.kpi .n { font-size: 18px; font-weight: 700; color: var(--accent); }
.kpi .l { color: var(--text-mute); font-size: 12px; margin-top: 4px; }
.hint { color: var(--text-mute); font-size: 12px; font-weight: 400; }
.card-head { display: flex; justify-content: space-between; align-items: baseline; }
.tag { background: var(--accent); color: white; padding: 2px 8px; border-radius: 8px; font-size: 11px; }
</style>
