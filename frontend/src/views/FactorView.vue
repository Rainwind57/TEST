<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useWatchlistStore } from '../stores/watchlist'
import { useToast } from '../stores/toast'
import api from '../api/client'
import { fmtPct, fmtNum, stripPrefix } from '../utils/format'

const store = useWatchlistStore()
const { toast } = useToast()
const router = useRouter()
const rows = ref([])
const loading = ref(false)
const catalog = ref([])
const activeGroup = ref('all')
// P3：表达式因子（选基础因子列用安全表达式合成新因子，落 user_factors）
const ufName = ref('')
const ufExpr = ref('')
const userFactors = ref([])

async function loadUserFactors() {
  try { userFactors.value = await api.get('/user-factors') } catch (e) { /* 未登录或无 uf */ }
}

async function saveExpressionFactor() {
  if (!ufName.value.trim()) { toast('请输入因子名称'); return }
  if (!ufExpr.value.trim()) { toast('请输入因子表达式'); return }
  try {
    const created = await api.post('/user-factors', {
      name: ufName.value.trim(), kind: 'expression', definition: { expression: ufExpr.value.trim() }
    })
    userFactors.value.unshift(created)
    ufName.value = ''; ufExpr.value = ''
    toast('表达式因子已保存，选股页以 uf:ID 引用')
  } catch (e) { toast(e.message) }
}

async function deleteExpressionFactor(id) {
  if (!confirm('删除该自定义因子？')) return
  try {
    await api.delete(`/user-factors/${id}`)
    userFactors.value = userFactors.value.filter(u => u.id !== id)
  } catch (e) { toast(e.message) }
}

const GROUP_LABELS = { all: '全部因子', technical: '量价技术', fundamental: '基本面', quant: '量能行情' }

const groupTabs = computed(() => {
  const groups = ['all', ...new Set(catalog.value.map(c => c.group))]
  return groups.map(g => ({ value: g, label: GROUP_LABELS[g] || g }))
})

const visibleCols = computed(() =>
  activeGroup.value === 'all' ? catalog.value : catalog.value.filter(c => c.group === activeGroup.value)
)

async function loadCatalog() {
  try {
    catalog.value = await api.get('/factors/catalog')
  } catch (e) {
    toast(e.message)
  }
}

async function runFactorTable() {
  if (!store.codes.length) { toast('请先在行情页添加自选股'); return }
  loading.value = true
  try {
    const data = await api.get('/factors', { params: { codes: store.codes.join(',') } })
    rows.value = data
  } catch (e) {
    toast(e.message)
  } finally {
    loading.value = false
  }
}

function formatCell(v, fmt) {
  if (v === null || v === undefined) return '--'
  if (fmt === 'pct') return fmtPct(v * 100)
  if (fmt === 'pct_raw') return fmtPct(v)
  return fmtNum(v, 2)
}

function cellCls(v, fmt) {
  if (v === null || v === undefined || (fmt !== 'pct' && fmt !== 'pct_raw')) return ''
  return v > 0 ? 'up' : v < 0 ? 'down' : 'flat'
}

onMounted(async () => {
  await loadCatalog()
  await loadUserFactors()
  if (!store.codes.length) await store.fetchWatchlist()
})
</script>

<template>
  <div>
    <div class="panel-toolbar">
      <button class="btn-primary" :disabled="loading" @click="runFactorTable">{{ loading ? '计算中…' : '计算自选股因子' }}</button>
      <button class="btn-ghost" @click="router.push('/backtest')">去回测</button>
      <span class="hint">技术因子基于腾讯历史K线（近260个交易日，前复权）计算；量能/基本面因子来自新浪实时行情快照，共 {{ catalog.length }} 个因子</span>
    </div>

    <div class="card">
      <div class="card-head"><h3>新建因子（表达式）</h3><span class="hint">用安全表达式对已有因子列合成新因子，保存后参与选股打分（选股页勾选 uf:ID）</span></div>
      <div class="panel-toolbar">
        <div class="field"><label>因子名称</label><input v-model="ufName" placeholder="如：动量增强" /></div>
        <div class="field grow"><label>表达式</label><input v-model="ufExpr" placeholder="momentum - volatility*0.5 ｜ 可用列：momentum/rsi/pe/turnover/mkt_cap… ｜ 函数：abs/log/sqrt/rank/zscore/clip" /></div>
        <button class="btn-primary" @click="saveExpressionFactor">保存因子</button>
      </div>
      <div v-if="userFactors.length" class="uf-list">
        <div v-for="u in userFactors" :key="u.id" class="uf-item">
          <span class="uf-name">{{ u.name }}</span>
          <span class="uf-detail">{{ u.kind === 'expression' ? u.definition?.expression : JSON.stringify(u.definition?.factors || []) }}</span>
          <button class="btn-ghost sm danger" @click="deleteExpressionFactor(u.id)">删除</button>
        </div>
      </div>
      <div v-else class="empty-hint">暂无自定义因子</div>
    </div>

    <div class="tabs">
      <button v-for="t in groupTabs" :key="t.value" class="tab-btn" :class="{ active: activeGroup === t.value }" @click="activeGroup = t.value">{{ t.label }}</button>
    </div>

    <div class="card">
      <div class="card-head"><h2>因子截面</h2><span class="count">{{ rows.length ? `共 ${rows.length} 只 · ${visibleCols.length} 个因子列` : '' }}</span></div>
      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th class="sticky-col">股票</th>
              <th v-for="col in visibleCols" :key="col.key">{{ col.label }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-if="!rows.length" class="empty-row"><td :colspan="visibleCols.length + 1">点击「计算自选股因子」开始</td></tr>
            <tr v-for="r in rows" :key="r.code">
              <template v-if="r.error">
                <td class="td-name sticky-col">{{ r.name }}<span class="td-code">{{ stripPrefix(r.code) }}</span></td>
                <td :colspan="visibleCols.length" class="empty-row">历史数据加载失败</td>
              </template>
              <template v-else>
                <td class="td-name sticky-col">{{ r.name }}<span class="td-code">{{ stripPrefix(r.code) }}</span></td>
                <td v-for="col in visibleCols" :key="col.key" :class="cellCls(r[col.key], col.format)">
                  {{ formatCell(r[col.key], col.format) }}
                </td>
              </template>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<style scoped>
.tabs { display: flex; gap: 8px; margin: 12px 0; flex-wrap: wrap; }
.tab-btn {
  padding: 6px 14px; border-radius: 6px; border: 1px solid var(--border);
  background: transparent; color: var(--text-mute); cursor: pointer; font-size: 13px;
}
.tab-btn.active { background: var(--accent); color: #fff; border-color: var(--accent); }
.table-scroll { overflow-x: auto; }
.table-scroll table { min-width: 100%; }
.sticky-col { position: sticky; left: 0; background: var(--card); z-index: 1; }
.panel-toolbar { display: flex; align-items: center; gap: 14px; margin: 12px 0; flex-wrap: wrap; }
.field { display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: var(--text-dim); }
.field.grow { flex: 1; min-width: 260px; }
.uf-list { margin-top: 12px; display: flex; flex-direction: column; gap: 8px; }
.uf-item { display: flex; align-items: center; gap: 10px; background: var(--bg-2); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; font-size: 13px; }
.uf-name { font-weight: 600; white-space: nowrap; }
.uf-detail { flex: 1; color: var(--text-mute); font-family: monospace; font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.btn-ghost.sm.danger { color: #ff6b6b; }
</style>
