<script setup>
import { ref, computed } from 'vue'
import api from '../api/client'
import { useToast } from '../stores/toast'
import EChart from '../components/EChart.vue'

const { toast } = useToast()

const loading = ref(false)
const result = ref(null)

const codesInput = ref('')
const weightsInput = ref('')

async function run() {
  loading.value = true
  try {
    result.value = await api.get('/risk/attribution')
    toast('持仓归因完成')
  } catch (e) { toast(e.message) }
  finally { loading.value = false }
}

async function runCustom() {
  const codes = codesInput.value.split(/[,，\s]+/).map(s => s.trim()).filter(Boolean)
  if (!codes.length) { toast('请输入组合代码，逗号分隔'); return }
  const weights = weightsInput.value.split(/[,，\s]+/).map(s => s.trim()).filter(Boolean)
  if (weights.length && weights.length !== codes.length) { toast('权重数量必须与代码数量一致'); return }
  loading.value = true
  try {
    result.value = await api.post('/risk/attribution', {
      codes,
      weights: weights.length ? weights.map(Number) : null,
      name: '自定义组合',
    })
    toast('自定义归因完成')
  } catch (e) { toast(e.message) }
  finally { loading.value = false }
}

const fmt = v => v == null ? '-' : Number(v).toFixed(4)
const fmtPct = v => v == null ? '-' : (Number(v) * 100).toFixed(2) + '%'

const contribOption = computed(() => {
  if (!result.value?.factorContribution) return {}
  const data = result.value.factorContribution.map((v, i) => ({
    name: result.value.factorNames[i], value: Number(v)
  }))
  return {
    tooltip: { trigger: 'item', formatter: '{b}: {c}' },
    grid: { left: 100, right: 20, top: 20, bottom: 20 },
    xAxis: { type: 'value' },
    yAxis: { type: 'category', data: data.map(d => d.name) },
    series: [{
      type: 'bar',
      data: data.map(d => d.value),
      itemStyle: { color: (p) => p.value >= 0 ? '#21c08b' : '#ff4d4f' }
    }]
  }
})
</script>

<template>
  <div>
    <div class="card">
      <div class="card-head"><h3>Barra 风格风险归因</h3><span class="hint">对当前模拟盘持仓做方差分解</span></div>
      <button class="btn-primary" :disabled="loading" @click="run">{{ loading ? '计算中…' : '运行归因(模拟盘持仓)' }}</button>
    </div>

    <div class="card">
      <div class="card-head"><h3>自定义组合归因</h3><span class="hint">传任意代码/权重组合，可复用选股结果或组合优化输出</span></div>
      <div class="custom-inputs">
        <div class="field grow"><label>代码（逗号分隔）</label><input v-model="codesInput" placeholder="如 600519,000001,300750" /></div>
        <div class="field"><label>权重（可选，与代码一一对应）</label><input v-model="weightsInput" placeholder="如 0.5,0.3,0.2，留空为等权" /></div>
      </div>
      <button class="btn-primary sm" :disabled="loading" @click="runCustom">{{ loading ? '计算中…' : '自定义归因' }}</button>
    </div>

    <div v-if="result" class="card">
      <h3>收益归因</h3>
      <div class="kpi-row">
        <div class="kpi"><div class="n">{{ fmtPct(result.totalReturn) }}</div><div class="l">组合收益</div></div>
        <div class="kpi"><div class="n">{{ fmtPct(result.factorReturn) }}</div><div class="l">因子贡献</div></div>
        <div class="kpi"><div class="n">{{ fmtPct(result.residual) }}</div><div class="l">残差(特质)</div></div>
      </div>
      <EChart :option="contribOption" style="height:220px" />
    </div>

    <div v-if="result" class="card">
      <h3>风险分解</h3>
      <div class="kpi-row">
        <div class="kpi"><div class="n">{{ fmtPct(result.risk?.factorRisk) }}</div><div class="l">因子风险</div></div>
        <div class="kpi"><div class="n">{{ fmtPct(result.risk?.specificRisk) }}</div><div class="l">特质风险</div></div>
        <div class="kpi"><div class="n">{{ fmtPct(result.risk?.totalRisk) }}</div><div class="l">总风险</div></div>
        <div class="kpi"><div class="n">{{ fmtPct(result.risk?.factorRiskPct) }}</div><div class="l">因子占比</div></div>
      </div>
      <table class="data-table">
        <thead><tr><th>代码</th><th>权重</th></tr></thead>
        <tbody>
          <tr v-for="h in result.holdings" :key="h.code">
            <td>{{ h.code }}</td><td class="up">{{ fmtPct(h.weight) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.data-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
.data-table th, .data-table td { border: 1px solid var(--border); padding: 8px 11px; text-align: left; }
.data-table th { background: var(--card-2); color: var(--text-dim); font-weight: 600; }
.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 14px 0; }
.kpi { background: var(--card-2); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }
.kpi .n { font-size: 18px; font-weight: 700; color: var(--accent); }
.kpi .l { color: var(--text-mute); font-size: 12px; margin-top: 4px; }
.hint { color: var(--text-mute); font-size: 12px; font-weight: 400; }
.card-head { display: flex; justify-content: space-between; align-items: baseline; }
.custom-inputs { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 12px; }
.custom-inputs .field { margin-bottom: 0; }
.custom-inputs .grow { flex: 2; min-width: 260px; }
.custom-inputs input { width: 100%; }
.btn-primary.sm { padding: 7px 16px; font-size: 13px; }
</style>
