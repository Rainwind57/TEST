<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import api from '../api/client'
import { useToast } from '../stores/toast'
import { useWatchlistStore } from '../stores/watchlist'
import { useResearchStore } from '../stores/research'
import { stripPrefix, fmtPct, fmtNum } from '../utils/format'
import Skeleton from '../components/Skeleton.vue'

const { toast } = useToast()
const watchlist = useWatchlistStore()
const router = useRouter()
const research = useResearchStore()

const BOARD_OPTIONS = [
  { value: 'all', label: '全部A股' },
  { value: 'sh_main', label: '沪市主板' },
  { value: 'sz_main', label: '深市主板' },
  { value: 'gem', label: '创业板' },
  { value: 'star', label: '科创板' },
  { value: 'bse', label: '北交所' }
]
const GROUP_LABELS = { quant: '量价类因子', fundamental: '估值/财务类因子', technical: '技术类因子（基于K线，计算较慢）', moneyflow: '资金流因子', custom: '自定义组合因子' }

const board = ref('all')
const poolSize = ref(200)
const topN = ref(20)
const loading = ref(false)
const applying = ref(false)
const result = ref(null)
const catalog = ref([])
const factorState = reactive({})
const checked = reactive(new Set())
const totalCash = ref(100000)

const filters = reactive({
  excludeSt: true, minPrice: null, maxPrice: null,
  minPe: null, maxPe: null, minMktCap: null, maxMktCap: null
})

const groupedCatalog = computed(() => {
  const groups = {}
  for (const f of catalog.value) {
    if (!groups[f.group]) groups[f.group] = []
    groups[f.group].push(f)
  }
  return groups
})

const selectedFactors = computed(() =>
  catalog.value
    .filter(f => factorState[f.key]?.checked)
    // direction=0 的因子（行业/宏观等）仅作信息展示，不参与加权打分
    .filter(f => Number(factorState[f.key].direction) !== 0)
    .map(f => ({ key: f.key, weight: Number(factorState[f.key].weight) || 1, direction: Number(factorState[f.key].direction) }))
)

const factorColumns = computed(() =>
  selectedFactors.value.map(s => catalog.value.find(f => f.key === s.key)).filter(Boolean)
)

async function loadCatalog() {
  const data = await api.get('/select/factors')
  // 加载当前用户自定义组合因子（key=uf:{id}，后端 run_select 已支持）
  let ufList = []
  try { ufList = await api.get('/user-factors') } catch (e) { /* 未登录或无 uf */ }
  const ufItems = (ufList || []).map(u => ({
    key: `uf:${u.id}`, label: u.name, group: 'custom',
    direction: 1, format: 'number', kline: false,
  }))
  catalog.value = [...data, ...ufItems]
  const defaults = ['momentum60', 'pe', 'turnover', 'momentum']
  for (const f of catalog.value) {
    factorState[f.key] = { checked: defaults.includes(f.key), weight: 1, direction: f.direction }
  }
}

function buildFilters() {
  const out = { excludeSt: filters.excludeSt }
  for (const k of ['minPrice', 'maxPrice', 'minPe', 'maxPe', 'minMktCap', 'maxMktCap']) {
    const v = filters[k]
    out[k] = (v === null || v === '' || v === undefined) ? null : Number(v)
  }
  return out
}

async function runSelect() {
  if (!selectedFactors.value.length) { toast('请至少勾选一个因子'); return }
  loading.value = true
  checked.clear()
  try {
    const data = await api.post('/select', {
      board: board.value, poolSize: Number(poolSize.value), topN: Number(topN.value),
      factors: selectedFactors.value, filters: buildFilters()
    })
    result.value = data
    research.setCurrentSelectResult({ board: board.value, poolSize: Number(poolSize.value), rows: data.rows })
  } catch (e) {
    toast(e.message)
    result.value = null
  } finally {
    loading.value = false
  }
}

function toggleCheck(code) {
  checked.has(code) ? checked.delete(code) : checked.add(code)
}
function toggleCheckAll() {
  if (!result.value) return
  if (checked.size === result.value.rows.length) checked.clear()
  else result.value.rows.forEach(r => checked.add(r.code))
}
function pickedCodes() {
  if (!result.value) return []
  return checked.size ? [...checked] : result.value.rows.map(r => r.code)
}

async function addToWatchlist() {
  const codes = pickedCodes()
  if (!codes.length) return
  applying.value = true
  try {
    const data = await api.post('/select/apply', { codes, action: 'watchlist' })
    toast(`已加入自选 ${data.added} 只`)
    await watchlist.fetchWatchlist()
  } catch (e) {
    toast(e.message)
  } finally {
    applying.value = false
  }
}

async function buyIntoPortfolio() {
  const codes = pickedCodes()
  if (!codes.length) return
  if (!totalCash.value || totalCash.value <= 0) { toast('请输入买入总资金'); return }
  applying.value = true
  try {
    const data = await api.post('/select/apply', { codes, action: 'buy', totalCash: Number(totalCash.value) })
    const okCount = data.results.filter(r => r.ok).length
    toast(`模拟盘买入完成：成功 ${okCount}/${data.results.length}`)
  } catch (e) {
    toast(e.message)
  } finally {
    applying.value = false
  }
}

// 把当前候选池参数带到主回测页（打通选股→回测，旧版选股结果只能加自选/买入）
function gotoBacktest() {
  research.setOptimalParams({ board: board.value, poolSize: Number(poolSize.value) })
  router.push('/backtest')
}

// 选股结果落盘为中间结果（artifact），供回测/组合/风险环节读取 codes 复用
const savingArtifact = ref(false)
const savedArtifactId = ref('')
async function saveArtifact() {
  if (!result.value) return
  savingArtifact.value = true
  try {
    const meta = await api.post('/artifacts', {
      kind: 'select',
      payload: {
        codes: result.value.rows.map(r => r.code),
        rows: result.value.rows,
        config: { board: board.value, poolSize: Number(poolSize.value), topN: Number(topN.value) },
      },
      name: `选股-${board.value}-${new Date().toLocaleString('zh-CN', { hour12: false })}`,
    })
    savedArtifactId.value = meta.id
    toast(`已保存中间结果：${meta.id}`)
  } catch (e) { toast(e.message) }
  finally { savingArtifact.value = false }
}

onMounted(loadCatalog)
</script>

<template>
  <div>
    <div class="card mb-24">
      <div class="card-head"><h2>候选池与筛选条件</h2></div>
      <div class="select-body">
        <div class="panel-row">
          <div class="field"><label>板块</label>
            <select v-model="board">
              <option v-for="b in BOARD_OPTIONS" :key="b.value" :value="b.value">{{ b.label }}</option>
            </select>
          </div>
          <div class="field"><label>候选池规模(按成交额取前N)</label><input v-model="poolSize" type="number" min="20" max="1000" /></div>
          <div class="field"><label>选出 TopN</label><input v-model="topN" type="number" min="1" max="100" /></div>
          <label class="checkbox-field"><input type="checkbox" v-model="filters.excludeSt" /> 剔除 ST</label>
        </div>
        <div class="panel-row">
          <div class="field"><label>价格区间</label>
            <div class="range-input"><input v-model="filters.minPrice" type="number" placeholder="最低" /><span>-</span><input v-model="filters.maxPrice" type="number" placeholder="最高" /></div>
          </div>
          <div class="field"><label>PE区间</label>
            <div class="range-input"><input v-model="filters.minPe" type="number" placeholder="最低" /><span>-</span><input v-model="filters.maxPe" type="number" placeholder="最高" /></div>
          </div>
          <div class="field"><label>市值区间(亿)</label>
            <div class="range-input"><input v-model="filters.minMktCap" type="number" placeholder="最低" /><span>-</span><input v-model="filters.maxMktCap" type="number" placeholder="最高" /></div>
          </div>
        </div>
      </div>
    </div>

    <div class="card mb-24">
      <div class="card-head"><h2>多因子加权打分</h2><span class="count">已选 {{ selectedFactors.length }} 个因子</span></div>
      <div class="factor-groups">
        <div v-for="(list, group) in groupedCatalog" :key="group" class="factor-group">
          <div class="group-title">{{ GROUP_LABELS[group] || group }}</div>
          <div class="factor-chip" v-for="f in list" :key="f.key">
            <label class="checkbox-field"><input type="checkbox" v-model="factorState[f.key].checked" /> {{ f.label }}</label>
            <input class="weight-input" type="number" step="0.1" v-model="factorState[f.key].weight" :class="{ dim: !factorState[f.key].checked }" title="权重（未勾选不参与，可提前编辑）" />
            <select class="direction-select" v-model.number="factorState[f.key].direction" :class="{ dim: !factorState[f.key].checked }">
              <option :value="1">正向</option>
              <option :value="-1">反向</option>
              <option v-if="factorState[f.key].direction === 0" :value="0">不参与</option>
            </select>
          </div>
        </div>
      </div>
      <div class="run-row">
        <button class="btn-primary" :disabled="loading" @click="runSelect">{{ loading ? '选股中…' : '运行选股' }}</button>
        <span class="hint">量价/估值因子来自全市场行情快照（快）；技术类因子需拉取K线（较慢，候选池越大耗时越长）</span>
      </div>
    </div>

    <div v-if="loading" class="card">
      <div class="card-head"><h2>选股结果</h2><span class="hint">计算中…</span></div>
      <Skeleton type="rows" :count="8" />
    </div>
    <div v-else-if="result" class="card">
      <div class="card-head">
        <h2>选股结果</h2>
        <span class="count">候选池 {{ result.candidateSize }}/{{ result.universeSize }} 只，取前 {{ result.rows.length }} 只</span>
      </div>
      <div class="apply-row">
        <button class="btn-ghost" @click="toggleCheckAll">{{ checked.size === result.rows.length ? '取消全选' : '全选' }}</button>
        <span class="hint">未勾选时默认对全部结果操作</span>
        <button class="btn-ghost" :disabled="applying" @click="addToWatchlist">加入自选</button>
        <input class="cash-input" v-model="totalCash" type="number" placeholder="买入总资金" />
        <button class="btn-primary" :disabled="applying" @click="buyIntoPortfolio">一键买入模拟盘(等权)</button>
        <button class="btn-ghost" @click="gotoBacktest">用该股池回测</button>
        <button class="btn-ghost" :disabled="savingArtifact" @click="saveArtifact">{{ savingArtifact ? '保存中…' : '保存为中间结果' }}</button>
        <span v-if="savedArtifactId" class="hint">{{ savedArtifactId }}</span>
      </div>
      <table>
        <thead>
          <tr>
            <th></th><th>排名</th><th>股票</th><th>现价</th><th>涨跌幅</th>
            <th v-for="f in factorColumns" :key="f.key">{{ f.label }}</th>
            <th>综合得分</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="r in result.rows" :key="r.code">
            <td><input type="checkbox" :checked="checked.has(r.code)" @change="toggleCheck(r.code)" /></td>
            <td>{{ r.rank }}</td>
            <td class="td-name">{{ r.name }}<span class="td-code">{{ stripPrefix(r.code) }}</span></td>
            <td>{{ fmtNum(r.price) }}</td>
            <td :class="r.pctChg > 0 ? 'up' : r.pctChg < 0 ? 'down' : 'flat'">{{ fmtPct(r.pctChg) }}</td>
            <td v-for="f in factorColumns" :key="f.key">
              {{ r.factorDetail?.[f.key]?.raw === null || r.factorDetail?.[f.key]?.raw === undefined ? '--' : fmtNum(r.factorDetail[f.key].raw) }}
            </td>
            <td>{{ fmtNum(r.score, 3) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.select-body { padding: 18px 20px; display: flex; flex-direction: column; gap: 14px; }
.panel-row { display: flex; align-items: end; gap: 14px; flex-wrap: wrap; }
.checkbox-field { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text-dim); white-space: nowrap; }
.checkbox-field input { width: auto; padding: 0; }
.range-input { display: flex; align-items: center; gap: 6px; }
.range-input input { width: 90px; }
.range-input span { color: var(--text-mute); }

.factor-groups { padding: 4px 20px 4px; display: flex; flex-direction: column; gap: 16px; }
.group-title { font-size: 12px; color: var(--text-mute); margin-bottom: 8px; }
.factor-chip {
  display: inline-flex; align-items: center; gap: 8px;
  background: var(--bg-2); border: 1px solid var(--border); border-radius: 8px;
  padding: 8px 10px; margin: 0 8px 8px 0;
}
.weight-input { width: 60px; padding: 6px 8px; }
.direction-select { width: 66px; padding: 6px 8px; }
.factor-chip .dim { opacity: .45; }
.run-row { display: flex; align-items: center; gap: 14px; padding: 16px 20px 20px; flex-wrap: wrap; }

.apply-row { display: flex; align-items: center; gap: 12px; padding: 16px 22px; border-bottom: 1px solid var(--border); flex-wrap: wrap; }
.cash-input { width: 140px; }
</style>
