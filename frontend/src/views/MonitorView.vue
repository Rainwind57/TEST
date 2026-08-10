<script setup>
import { ref, computed, onMounted } from 'vue'
import api from '../api/client'
import { useToast } from '../stores/toast'
import EChart from '../components/EChart.vue'

const { toast } = useToast()

const enabled = ref(false)
const lastRun = ref(null)
const signals = ref([])
const equity = ref([])
const loading = ref(false)
// P5：信号引擎配置（rule=内置规则 | model=落盘 ML 模型）
const mode = ref('rule')
const modelId = ref('')
const modelOptions = ref([])
const savingCfg = ref(false)
const scanning = ref(false)
// 盯盘评分口径：isolated=孤立打分 / full=全池排名分位（与选股口径一致）
const ranking = ref('isolated')
// 自动调仓开关：默认关闭，开启后信号触发自动买卖
const autoTrade = ref(false)
const savingAutoTrade = ref(false)

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
    }
  } catch (e) { toast(e.message) }
}

async function loadAutoTrade() {
  try {
    const r = await api.get('/monitor/auto-trade')
    autoTrade.value = !!r.autoTrade
  } catch (e) { /* 静默，旧后端可能无此接口 */ }
}

async function loadModels() {
  try { modelOptions.value = await api.get('/ml/models') }
  catch (e) { /* 静默 */ }
}

async function saveConfig() {
  if (mode.value === 'model' && !modelId.value) { toast('模型模式下请先选择模型'); return }
  savingCfg.value = true
  try {
    const cfg = await api.post('/monitor/config', {
      mode: mode.value,
      modelId: modelId.value,
      ranking: ranking.value
    })
    toast(cfg.mode === 'model' ? `信号引擎已切换为模型 ${cfg.modelId}（口径：${cfg.ranking === 'full' ? '全池排名' : '孤立打分'}）` : '信号引擎已切换为内置规则')
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

// 手动立即扫描：无需等交易日 15:10 的 cron，随时验证盯盘能否产出信号
async function scanNow() {
  scanning.value = true
  try {
    const res = await api.post('/monitor/scan?force=true')
    signals.value = res.signals || []
    toast(res.ok ? `扫描完成，共 ${signals.value.length} 条信号` : res.reason || '扫描被跳过')
    await loadStatus()
  } catch (e) { toast(e.message) }
  finally { scanning.value = false }
}

async function refresh() {
  await Promise.all([loadStatus(), loadEquity(), loadAutoTrade()])
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

const fmt = v => v == null ? '-' : Number(v).toFixed(3)
const fmtPct = v => v == null ? '-' : (Number(v) * 100).toFixed(2) + '%'
const fmtTime = t => t ? t.slice(0, 19).replace('T', ' ') : '-'

onMounted(() => { refresh(); loadModels() })
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
      <h3>盯盘信号（全量扫描）</h3>
      <div v-if="!signals.length" class="empty-hint">暂无信号，调度器运行后 15:10 生成</div>
      <table v-else class="data-table">
        <thead><tr><th>代码</th><th>引擎</th><th>动量</th><th>RSI</th><th>模型得分</th><th>信号</th></tr></thead>
        <tbody>
          <tr v-for="s in signals" :key="s.code">
            <td>{{ s.code }}</td>
            <td>{{ s.mode === 'model' ? 'ML模型' : '规则' }}</td>
            <td>{{ s.momentum == null ? '-' : fmtPct(s.momentum) }}</td>
            <td>{{ s.rsi == null ? '-' : fmt(s.rsi) }}</td>
            <td :class="(s.score || 0) > 0 ? 'up' : 'down'">{{ s.score == null ? '-' : fmt(s.score) }}</td>
            <td><span class="tag" v-if="s.signal">{{ s.signal }}</span><span v-else class="muted">-</span></td>
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
.data-table th, .data-table td { border: 1px solid var(--border); padding: 9px 11px; text-align: left; }
.data-table th { background: var(--card-2); color: var(--text-dim); font-weight: 600; }
.warn-tag { background: #ffb454; color: #1a1a2e; padding: 2px 10px; border-radius: 10px; font-size: 12px; font-weight: 600; }
.warn-box { background: rgba(255,180,84,.12); border: 1px solid #ffb45455; border-radius: 8px; padding: 10px 14px; font-size: 13px; color: #7a5200; }
.status-row { margin-top: 12px; display: flex; gap: 24px; font-size: 13px; color: var(--text-dim); }
.hint { color: var(--text-mute); font-size: 12px; }
.card-head { display: flex; justify-content: space-between; align-items: baseline; }
.tag { background: var(--accent); color: white; padding: 2px 8px; border-radius: 8px; font-size: 12px; }
.muted { color: var(--text-mute); }
.empty-hint { color: var(--text-mute); font-size: 13px; padding: 16px 0; }
.auto-trade-row { margin-top: 12px; display: flex; align-items: center; gap: 12px; font-size: 13px; }
.switch-label { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; user-select: none; }
.switch-label input { width: 16px; height: 16px; cursor: pointer; }
</style>
