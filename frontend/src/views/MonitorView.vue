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

async function loadStatus() {
  try {
    const s = await api.get('/monitor/status')
    enabled.value = s.enabled
    lastRun.value = s.lastRun
    signals.value = s.signals || []
  } catch (e) { toast(e.message) }
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

async function refresh() {
  await Promise.all([loadStatus(), loadEquity()])
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

onMounted(refresh)
</script>

<template>
  <div>
    <div class="card">
      <div class="card-head"><h3>盯盘调度器</h3>
        <span class="warn-tag" v-if="enabled">运行中 · 模拟、非实盘</span>
        <span class="hint" v-else>已关闭</span>
      </div>
      <div class="warn-box">
        自动调仓仅限模拟盘，默认开启（重启保持状态）。交易日历基于K线数据动态推断，非周末且不在节假日则执行。
      </div>
      <div class="panel-toolbar" style="margin-top:10px">
        <button class="btn-primary" :disabled="loading" @click="toggle">
          {{ enabled ? '停止调度' : '开启调度' }}
        </button>
        <button class="btn-ghost" @click="refresh">刷新</button>
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
        <thead><tr><th>代码</th><th>动量</th><th>RSI</th><th>信号</th></tr></thead>
        <tbody>
          <tr v-for="s in signals" :key="s.code">
            <td>{{ s.code }}</td>
            <td class="up">{{ fmtPct(s.momentum) }}</td>
            <td>{{ fmt(s.rsi) }}</td>
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
</style>
