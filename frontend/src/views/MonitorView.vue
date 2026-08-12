<script setup>
import { ref, computed, onMounted } from 'vue'
import api from '../api/client'
import { useToast } from '../stores/toast'
import EChart from '../components/EChart.vue'
import BoardSelect from '../components/BoardSelect.vue'

const { toast } = useToast()

const enabled = ref(false)
const lastRun = ref(null)
const signals = ref([])
const equity = ref([])
const loading = ref(false)
const mode = ref('rule')
const modelId = ref('')
const modelOptions = ref([])
const savingCfg = ref(false)
const scanning = ref(false)
const ranking = ref('isolated')
const autoTrade = ref(false)
const savingAutoTrade = ref(false)
const monitorConfig = ref(null)

// 模型模式下的板块/池规模/调参（与选股口径对齐）
const board = ref('all')
const poolSize = ref(150)
const adjustId = ref('')
const adjustOptions = ref([])
const addWatchlistBusy = ref({})
const buyBusy = ref({})

async function loadStatus() {
  try {
    const s = await api.get('/monitor/status')
    enabled.value = s.enabled
    lastRun.value = s.lastRun
    signals.value = s.signals || []
    if (s.config) {
      mode.value = s.config.mode || 'rule'
      modelId.value = s.config.modelId || ''
      ranking.value = s.config.ranking || 'isolated'
      board.value = s.config.board || 'all'
      poolSize.value = s.config.poolSize || 150
      adjustId.value = s.config.adjustId || ''
      monitorConfig.value = s.config
    }
  } catch (e) { toast(e.message) }
}

async function loadAutoTrade() {
  try {
    const r = await api.get('/monitor/auto-trade')
    autoTrade.value = !!r.autoTrade
  } catch (e) { /* 静默 */ }
}

async function loadModels() {
  try { modelOptions.value = await api.get('/ml/models') }
  catch (e) { /* 静默 */ }
}

async function loadAdjusts() {
  try {
    const items = await api.get('/artifacts', { params: { kind: 'ml_adjust', limit: 50 } })
    adjustOptions.value = (items || []).map(a => ({
      id: a.id, name: a.name || a.id, modelId: a.payload?.modelId || ''
    }))
  } catch (e) { /* 静默 */ }
}

async function saveConfig() {
  if (mode.value === 'model' && !modelId.value) { toast('模型模式下请先选择模型'); return }
  savingCfg.value = true
  try {
    const cfg = await api.post('/monitor/config', {
      mode: mode.value,
      modelId: modelId.value,
      ranking: ranking.value,
      board: board.value,
      poolSize: Number(poolSize.value),
      adjustId: adjustId.value,
    })
    monitorConfig.value = cfg
    const desc = cfg.mode === 'model'
      ? `模型 ${cfg.modelId} · ${cfg.ranking === 'full' ? '全池排名' : '孤立打分'} · ${cfg.board}(${cfg.poolSize}只)`
      : '内置规则（动量+RSI）'
    toast(`信号引擎已切换：${desc}`)
  } catch (e) { toast(e.message) }
  finally { savingCfg.value = false }
}

async function toggleAutoTrade() {
  savingAutoTrade.value = true
  try {
    const r = await api.post('/monitor/auto-trade', { enabled: autoTrade.value })
    autoTrade.value = r.autoTrade
    toast(autoTrade.value ? '自动调仓已开启：信号将自动生成买卖单（模拟盘）' : '自动调仓已关闭')
  } catch (e) { toast(e.message); autoTrade.value = !autoTrade.value }
  finally { savingAutoTrade.value = false }
}

async function loadEquity() {
  try { equity.value = await api.get('/monitor/equity?limit=90') }
  catch (e) { toast(e.message) }
}

async function toggle() {
  loading.value = true
  try {
    const res = await api.post('/monitor/toggle', { enabled: !enabled.value })
    enabled.value = res.enabled
    toast(res.enabled ? '调度器已开启（交易日 15:05/15:10 执行）' : '调度器已关闭')
  } catch (e) { toast(e.message) }
  finally { loading.value = false }
}

async function scanNow() {
  scanning.value = true
  try {
    toast('正在扫描全市场K线数据，约需30-60秒，请稍候...')
    const res = await api.post('/monitor/scan?force=true')
    signals.value = res.signals || []
    toast(res.ok ? `扫描完成，共 ${signals.value.length} 条信号` : res.reason || '扫描被跳过')
    await loadStatus()
  } catch (e) { toast(e.message || '扫描超时，请稍后重试') }
  finally { scanning.value = false }
}

async function refresh() {
  await Promise.all([loadStatus(), loadEquity(), loadAutoTrade()])
}

async function addToWatchlist(code, name) {
  addWatchlistBusy.value[code] = true
  try {
    await api.post('/watchlist', { code, name: name || code })
    toast(`${code} 已加入自选`)
  } catch (e) { toast(e.message) }
  finally { addWatchlistBusy.value[code] = false }
}

async function buySignal(code, name) {
  buyBusy.value[code] = true
  try {
    await api.post('/portfolio/order', { code, name: name || code, side: 'buy', qty: 100 })
    toast(`${code} 已下单买入100股`)
  } catch (e) { toast(e.message) }
  finally { buyBusy.value[code] = false }
}

async function swapToSignal(s) {
  try {
    // 卖出原持仓100股 + 买入换仓目标100股
    await api.post('/portfolio/order', { code: s.code, name: s.name || s.code, side: 'sell', qty: 100 })
    await api.post('/portfolio/order', { code: s.swapTo, name: s.swapToName || s.swapTo, side: 'buy', qty: 100 })
    toast(`已换仓：卖出 ${s.code} → 买入 ${s.swapTo}`)
  } catch (e) { toast(e.message) }
}

async function batchAddWatchlist() {
  const codes = signals.value.filter(s => s.signal).map(s => s.code)
  if (!codes.length) { toast('无可用信号'); return }
  let ok = 0
  for (const s of signals.value) {
    if (!s.signal) continue
    try {
      await api.post('/watchlist', { code: s.code, name: s.name || s.code })
      ok++
    } catch (e) { /* skip */ }
  }
  toast(`已批量加入自选：${ok}/${codes.length}`)
}

async function batchBuy() {
  const actionable = signals.value.filter(s =>
    s.signal && (s.signal.includes('看多') || s.signal.includes('突破'))
  )
  if (!actionable.length) { toast('无看多信号可买入'); return }
  let ok = 0
  for (const s of actionable) {
    try {
      await api.post('/portfolio/order', { code: s.code, name: s.name || s.code, side: 'buy', qty: 100 })
      ok++
    } catch (e) { /* skip */ }
  }
  toast(`已批量下单买入：${ok}/${actionable.length}`)
}

const equityOption = computed(() => {
  const data = equity.value.map(e => [e.ts.slice(0, 16), Number(e.value)])
  return {
    tooltip: { trigger: 'axis' },
    grid: { left: 60, right: 20, top: 30, bottom: 40 },
    xAxis: { type: 'category', data: data.map(d => d[0]) },
    yAxis: { type: 'value', name: '净值' },
    series: [{ type: 'line', data: data.map(d => d[1]), smooth: true,
      itemStyle: { color: '#4f8cff' }, areaStyle: { opacity: 0.15 } }]
  }
})

const signalClass = sig => {
  if (!sig) return 'muted'
  if (sig.includes('看多') || sig.includes('突破')) return 'up'
  if (sig.includes('看空') || sig.includes('减仓') || sig.includes('平仓') || sig.includes('走弱')) return 'down'
  return 'muted'
}
const signalStyle = sig => {
  if (!sig) return {}
  if (sig.includes('持仓')) return { background: '#e6a81722', borderColor: '#e6a817', color: '#e6a817' }
  if (sig.includes('看多') || sig.includes('突破')) return { background: '#00c85322', borderColor: '#00c853', color: '#00c853' }
  if (sig.includes('看空')) return { background: '#ff525222', borderColor: '#ff5252', color: '#ff5252' }
  if (sig.includes('平仓') || sig.includes('减仓')) return { background: '#ff525222', borderColor: '#ff5252', color: '#ff5252' }
  return { background: '#e6a81722', borderColor: '#e6a817', color: '#e6a817' }
}

const fmt = v => v == null ? '-' : Number(v).toFixed(3)
const fmtPct = v => v == null ? '-' : (Number(v) * 100).toFixed(2) + '%'
const fmtTime = t => t ? t.slice(0, 19).replace('T', ' ') : '-'

const signalStats = computed(() => {
  let bullish = 0, bearish = 0, neutral = 0, position = 0
  signals.value.forEach(s => {
    if (s.inPosition) position++
    if (!s.signal) neutral++
    else if (s.signal.includes('看多') || s.signal.includes('突破')) bullish++
    else if (s.signal.includes('看空') || s.signal.includes('平仓') || s.signal.includes('减仓') || s.signal.includes('走弱') || s.signal.includes('超买')) bearish++
    else neutral++
  })
  return { bullish, bearish, neutral, position, total: signals.value.length }
})

const lastScanTime = computed(() => {
  if (!lastRun.value || !lastRun.value.ts) return null
  return fmtTime(lastRun.value.ts)
})

onMounted(() => { refresh(); loadModels(); loadAdjusts() })
</script>

<template>
  <div>
    <div class="card">
      <div class="card-head"><h3>盯盘调度器</h3>
        <span class="warn-tag" v-if="enabled">运行中 · 模拟、非实盘</span>
        <span class="hint" v-else>已关闭</span>
      </div>
      <div class="warn-box">
        自动调仓仅限模拟盘，默认关闭。交易日历基于K线数据动态推断，非周末且不在节假日则执行。
      </div>
      <div class="status-row" v-if="monitorConfig" style="margin-top:8px">
        <span class="hint">
          当前引擎：<strong>{{ monitorConfig.mode === 'model' ? 'ML模型' : '内置规则（动量+RSI）' }}</strong>
          <template v-if="monitorConfig.mode === 'model'">
            · 排名口径：<strong>{{ monitorConfig.ranking === 'full' ? '全池排名分位（与选股一致）' : '孤立打分' }}</strong>
            · 板块：<strong>{{ monitorConfig.board || 'all' }}</strong>
            · 池规模：<strong>{{ monitorConfig.poolSize || 150 }}</strong>
            <span v-if="monitorConfig.adjustId"> · 调参：<strong>{{ monitorConfig.adjustId }}</strong></span>
          </template>
        </span>
      </div>
      <div class="panel-toolbar" style="margin-top:10px">
        <div class="field"><label>信号引擎</label>
          <select v-model="mode">
            <option value="rule">内置规则（动量+RSI）</option>
            <option value="model">ML模型打分</option>
          </select>
        </div>
        <div v-if="mode === 'model'" class="field"><label>模型</label>
          <select v-model="modelId">
            <option value="">请选择模型</option>
            <option v-for="m in modelOptions" :key="m.id" :value="m.id">{{ m.id }}</option>
          </select>
        </div>
        <div v-if="mode === 'model'" class="field"><label>评分口径</label>
          <select v-model="ranking" :title="ranking === 'full' ? '全池排名分位，与选股口径一致' : '各股孤立打分'">
            <option value="isolated">孤立打分</option>
            <option value="full">全池排名分位</option>
          </select>
        </div>
        <div v-if="mode === 'model'" class="field"><label>板块</label>
          <BoardSelect v-model="board" style="min-width:100px" />
        </div>
        <div v-if="mode === 'model'" class="field"><label>池规模</label>
          <input v-model.number="poolSize" type="number" min="30" max="500" step="10" class="num-inp" />
        </div>
        <div v-if="mode === 'model' && adjustOptions.length" class="field"><label>调参</label>
          <select v-model="adjustId">
            <option value="">不使用调参</option>
            <option v-for="a in adjustOptions" :key="a.id" :value="a.id">{{ a.name }}</option>
          </select>
        </div>
        <button class="btn-ghost" :disabled="savingCfg" @click="saveConfig">{{ savingCfg ? '保存中…' : '保存配置' }}</button>
        <button class="btn-ghost" :disabled="scanning" @click="scanNow">{{ scanning ? '扫描中…' : '立即扫描' }}</button>
        <button class="btn-primary" :disabled="loading" @click="toggle">
          {{ enabled ? '停止调度' : '开启调度' }}
        </button>
        <button class="btn-ghost" @click="refresh">刷新</button>
      </div>
      <div class="auto-trade-row">
        <label class="switch-label">
          <input type="checkbox" v-model="autoTrade" :disabled="savingAutoTrade" @change="toggleAutoTrade" />
          <span>自动调仓</span>
        </label>
        <span class="hint" v-if="autoTrade">已开启：信号将自动生成买卖单（模拟盘）</span>
        <span class="hint" v-else>未开启：仅生成信号，不自动下单</span>
      </div>
      <div v-if="lastRun" class="status-row">
        <span>上次执行：{{ lastRun.task }} · {{ fmtTime(lastRun.ts) }}</span>
        <span v-if="lastRun.totalValue != null">总市值 {{ fmt(lastRun.totalValue/10000) }}万</span>
        <span v-if="lastRun.error" class="down">{{ lastRun.error }}</span>
      </div>
    </div>

    <div class="card">
      <div class="card-head">
        <h3>盯盘信号（全量扫描）</h3>
        <div class="signal-summary" v-if="signals.length">
          <span class="stat-badge up">🟢 {{ signalStats.bullish }} 看多</span>
          <span class="stat-badge down">🔴 {{ signalStats.bearish }} 看空/减仓</span>
          <span class="stat-badge muted">🟡 {{ signalStats.neutral }} 中性</span>
          <span v-if="signalStats.position" class="stat-badge pos">📌 {{ signalStats.position }} 持仓</span>
          <span class="hint" style="margin-left:4px">{{ lastScanTime ? '扫描于 ' + lastScanTime : '' }}</span>
        </div>
        <div v-if="signals.length" class="batch-actions">
          <button class="btn-ghost sm" @click="batchAddWatchlist">📋 全部加入自选</button>
          <button class="btn-ghost sm" @click="batchBuy">💰 看多信号批量买入</button>
        </div>
      </div>
      <div v-if="!signals.length" class="empty-hint">
        暂无信号。请先确保自选股列表不为空（在选股页面添加），然后切换信号引擎并点击"立即扫描"。
      </div>
      <table v-else class="data-table">
        <thead><tr>
          <th>代码</th><th>名称</th><th>引擎</th><th>动量</th><th>RSI</th><th>模型得分</th><th>信号</th><th>操作</th>
        </tr></thead>
        <tbody>
          <tr v-for="s in signals" :key="s.code" :class="{ 'row-in-position': s.inPosition }">
            <td><code>{{ s.code }}</code></td>
            <td>{{ s.name || '-' }}</td>
            <td><span class="tiny-tag">{{ s.mode === 'model' ? 'ML' : '规则' }}</span></td>
            <td>{{ s.momentum == null ? '-' : fmtPct(s.momentum) }}</td>
            <td>{{ s.rsi == null ? '-' : fmt(s.rsi) }}</td>
            <td :class="(s.score || 0) > 0 ? 'up' : 'down'">{{ s.score == null ? '-' : fmt(s.score) }}</td>
            <td>
              <span v-if="s.signal" class="signal-badge" :style="signalStyle(s.signal)">{{ s.signal }}</span>
              <span v-else class="muted">-</span>
              <div v-if="s.swapTo" class="swap-hint">🔄 建议换为 <code>{{ s.swapTo }}</code>（得分 {{ fmt(s.swapScore) }}）</div>
            </td>
            <td class="actions-cell">
              <button class="btn-mini" :disabled="addWatchlistBusy[s.code]" @click="addToWatchlist(s.code, s.name)" title="加入自选">⭐</button>
              <button class="btn-mini" :disabled="buyBusy[s.code]" @click="buySignal(s.code, s.name)" title="模拟买入100股">💰</button>
              <button v-if="s.swapTo" class="btn-mini" @click="swapToSignal(s)" title="卖出100股换仓">🔄</button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="card">
      <h3>净值曲线（近90条）</h3>
      <EChart :option="equityOption" style="height:320px" />
    </div>
  </div>
</template>

<style scoped>
.data-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
.data-table th, .data-table td { border: 1px solid var(--border); padding: 7px 9px; text-align: left; }
.data-table th { background: var(--card-2); color: var(--text-dim); font-weight: 600; white-space: nowrap; }
.warn-tag { background: #ffb454; color: #1a1a2e; padding: 2px 10px; border-radius: 10px; font-size: 12px; font-weight: 600; }
.warn-box { background: rgba(255,180,84,.12); border: 1px solid #ffb45455; border-radius: 8px; padding: 10px 14px; font-size: 13px; color: #c3860a; }
.status-row { margin-top: 12px; display: flex; gap: 24px; font-size: 13px; color: var(--text-dim); flex-wrap: wrap; }
.hint { color: var(--text-mute); font-size: 12px; }
.card-head { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px; }
.tag { background: var(--accent); color: white; padding: 2px 8px; border-radius: 8px; font-size: 12px; }
.tiny-tag { font-size: 10px; padding: 1px 5px; border-radius: 4px; background: var(--card-2); color: var(--text-dim); }
.muted { color: var(--text-mute); }
.empty-hint { color: var(--text-mute); font-size: 13px; padding: 16px 0; }
.auto-trade-row { margin-top: 12px; display: flex; align-items: center; gap: 12px; font-size: 13px; }
.switch-label { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; user-select: none; }
.switch-label input { width: 16px; height: 16px; cursor: pointer; }
.num-inp { width: 70px; padding: 4px 6px; border: 1px solid var(--border); border-radius: 4px; background: var(--bg); color: var(--text); }
.signal-badge { display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 12px; border: 1px solid; font-weight: 500; white-space: nowrap; }
.actions-cell { white-space: nowrap; }
.btn-mini { padding: 2px 6px; font-size: 13px; border-radius: 4px; border: 1px solid var(--border); background: var(--card-2); cursor: pointer; margin-right: 4px; }
.btn-mini:hover { background: var(--accent); color: white; }
.btn-mini:disabled { opacity: 0.4; cursor: not-allowed; }
.row-in-position { background: rgba(230,168,23,.06); }
.batch-actions { display: flex; gap: 6px; }
.btn-ghost.sm { font-size: 12px; padding: 3px 10px; }
.down { color: #ff5252; }
.signal-summary { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; font-size: 13px; }
.stat-badge { padding: 2px 10px; border-radius: 10px; font-size: 12px; font-weight: 500; }
.stat-badge.up { background: #00c85322; color: #00c853; border: 1px solid #00c85344; }
.stat-badge.down { background: #ff525222; color: #ff5252; border: 1px solid #ff525244; }
.stat-badge.muted { background: var(--card-2); color: var(--text-dim); border: 1px solid var(--border); }
.stat-badge.pos { background: #e6a81722; color: #e6a817; border: 1px solid #e6a81744; }
.swap-hint { font-size: 11px; color: var(--accent); margin-top: 3px; }
</style>
