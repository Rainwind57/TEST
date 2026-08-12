<script setup>
import { ref, reactive, onMounted, computed } from 'vue'
import api from '../api/client'
import { useToast } from '../stores/toast'

const { toast } = useToast()

const strategies = ref([])
const userFactors = ref([])
const runs = ref([])
const catalog = ref([])
const loading = ref(false)
const running = ref(null)

const ufName = ref('')
const ufRows = reactive([{ key: 'momentum', weight: 1, direction: 1 }])

const techFactors = computed(() => catalog.value.filter(f => f.kline))
const snapFactors = computed(() => catalog.value.filter(f => !f.kline))

async function loadAll() {
  loading.value = true
  try {
    const [s, u, r, c] = await Promise.all([
      api.get('/strategies'), api.get('/user-factors'),
      api.get('/backtest-runs'), api.get('/factors/catalog')
    ])
    strategies.value = s; userFactors.value = u; runs.value = r; catalog.value = c
  } catch (e) { toast(e.message) }
  finally { loading.value = false }
}

async function runStrategy(id) {
  running.value = id
  try {
    const res = await api.post(`/strategies/${id}/run`)
    if (res.metrics) {
      await api.post('/backtest-runs', { strategyId: id, config: res.config, metrics: res.metrics })
      toast(`运行完成，年化 ${(res.metrics.annualizedReturn * 100).toFixed(1)}%，已存档`)
      runs.value = await api.get('/backtest-runs')
    } else {
      toast('运行完成')
    }
  } catch (e) { toast(e.message) }
  finally { running.value = null }
}

async function deleteStrategy(id) {
  if (!confirm('删除该策略？')) return
  try { await api.delete(`/strategies/${id}`); strategies.value = strategies.value.filter(s => s.id !== id) }
  catch (e) { toast(e.message) }
}

function addUfRow() { ufRows.push({ key: 'momentum', weight: 1, direction: 1 }) }
function removeUfRow(i) { if (ufRows.length > 1) ufRows.splice(i, 1) }

async function saveUserFactor() {
  if (!ufName.value.trim()) { toast('请输入因子名称'); return }
  const factors = ufRows.map(r => ({ key: r.key, weight: Number(r.weight), direction: Number(r.direction) }))
  try {
    const created = await api.post('/user-factors', { name: ufName.value.trim(), kind: 'composite', definition: { factors } })
    userFactors.value.unshift(created)
    ufName.value = ''; ufRows.splice(0, ufRows.length, { key: 'momentum', weight: 1, direction: 1 })
    toast('自定义因子已保存')
  } catch (e) { toast(e.message) }
}

async function deleteUserFactor(id) {
  if (!confirm('删除该自定义因子？')) return
  try { await api.delete(`/user-factors/${id}`); userFactors.value = userFactors.value.filter(u => u.id !== id) }
  catch (e) { toast(e.message) }
}

async function deleteRun(id) {
  if (!confirm('删除该回测记录？')) return
  try { await api.delete(`/backtest-runs/${id}`); runs.value = runs.value.filter(r => r.id !== id) }
  catch (e) { toast(e.message) }
}

const fmtTime = t => t ? t.slice(0, 19).replace('T', ' ') : '-'
const fmtPct = v => v == null ? '-' : (Number(v) * 100).toFixed(2) + '%'
const fmtNum = v => v == null ? '-' : Number(v).toFixed(3)

onMounted(loadAll)
</script>

<template>
  <div>
    <div class="card">
      <div class="card-head"><h3>已保存策略</h3><span class="hint">一键重跑已持久化的选股/回测配置</span></div>
      <div v-if="!strategies.length" class="empty-hint">暂无策略，前往「选股/分层回测」页保存</div>
      <table v-else class="data-table">
        <thead><tr><th>名称</th><th>类型</th><th>关键配置</th><th>创建时间</th><th>操作</th></tr></thead>
        <tbody>
          <tr v-for="s in strategies" :key="s.id">
            <td>{{ s.name }}</td>
            <td>{{ s.kind }}</td>
            <td class="muted">{{ s.config?.factor || (s.config?.factors?.length ? s.config.factors.length + '因子' : '-') }}</td>
            <td>{{ fmtTime(s.created_at) }}</td>
            <td>
              <button class="btn-ghost sm" :disabled="running === s.id" @click="runStrategy(s.id)">{{ running === s.id ? '运行中…' : '运行' }}</button>
              <button class="btn-ghost sm danger" @click="deleteStrategy(s.id)">删除</button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="card">
      <div class="card-head"><h3>自定义因子（组合式）</h3><span class="hint">对已有因子加权 z-score 合成新因子，选股时以 uf:ID 引用</span></div>
      <div class="panel-toolbar" style="margin-bottom:10px">
        <div class="field"><label>因子名称</label><input v-model="ufName" placeholder="如：动量+RSI复合" /></div>
      </div>
      <div v-for="(row, i) in ufRows" :key="i" class="panel-toolbar uf-row">
        <div class="field">
          <label>成分{{ i + 1 }}</label>
          <select v-model="row.key">
            <optgroup label="量价因子">
              <option v-for="f in techFactors" :key="f.key" :value="f.key">{{ f.label }}</option>
            </optgroup>
            <optgroup label="快照因子">
              <option v-for="f in snapFactors" :key="f.key" :value="f.key">{{ f.label }}</option>
            </optgroup>
          </select>
        </div>
        <div class="field"><label>权重</label><input v-model="row.weight" type="number" step="0.1" /></div>
        <div class="field"><label>方向</label>
          <select v-model="row.direction"><option :value="1">正向</option><option :value="-1">反向</option></select>
        </div>
        <button class="btn-ghost sm" @click="removeUfRow(i)">移除</button>
      </div>
      <div class="panel-toolbar" style="margin-top:8px">
        <button class="btn-ghost" @click="addUfRow">+ 添加成分</button>
        <button class="btn-primary" @click="saveUserFactor">保存自定义因子</button>
      </div>
      <div v-if="!userFactors.length" class="empty-hint">暂无自定义因子</div>
      <table v-else class="data-table">
        <thead><tr><th>名称</th><th>成分</th><th>创建时间</th><th>操作</th></tr></thead>
        <tbody>
          <tr v-for="u in userFactors" :key="u.id">
            <td>{{ u.name }}</td>
            <td class="muted">{{ (u.definition?.factors || []).map(f => f.key + '(' + f.weight + ')').join(' + ') }}</td>
            <td>{{ fmtTime(u.created_at) }}</td>
            <td><button class="btn-ghost sm danger" @click="deleteUserFactor(u.id)">删除</button></td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="card">
      <div class="card-head"><h3>回测存档</h3><span class="hint">每次一键运行策略自动归档关键指标</span></div>
      <div v-if="!runs.length" class="empty-hint">暂无回测记录</div>
      <table v-else class="data-table">
        <thead><tr><th>时间</th><th>年化收益</th><th>Sharpe</th><th>最大回撤</th><th>胜率</th><th>操作</th></tr></thead>
        <tbody>
          <tr v-for="r in runs" :key="r.id">
            <td>{{ fmtTime(r.created_at) }}</td>
            <td class="up">{{ fmtPct(r.metrics?.annualizedReturn) }}</td>
            <td>{{ fmtNum(r.metrics?.sharpe) }}</td>
            <td class="down">{{ fmtPct(r.metrics?.maxDrawdown) }}</td>
            <td>{{ fmtPct(r.metrics?.winRate) }}</td>
            <td><button class="btn-ghost sm danger" @click="deleteRun(r.id)">删除</button></td>
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
.data-table .muted { color: var(--text-mute); font-size: 12px; }
.btn-ghost.sm { padding: 4px 10px; font-size: 12px; margin-right: 6px; }
.btn-ghost.sm.danger { color: #ff6b6b; }
.hint { color: var(--text-mute); font-size: 12px; font-weight: 400; }
.uf-row { padding: 6px 0; margin: 0 !important; border-bottom: 1px dashed var(--border); }
.card-head { display: flex; justify-content: space-between; align-items: baseline; }
</style>
