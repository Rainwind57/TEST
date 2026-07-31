<script setup>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import api from '../api/client'
import { useToast } from '../stores/toast'
import EChart from '../components/EChart.vue'

const { toast } = useToast()
const router = useRouter()

const codesText = ref('sh600519,sz000001,sz300750,sh601318,sh600000')
const muText = ref('0.10,0.05,0.08,0.12,0.03')
const covText = ref('0.04,0.01,0.01,0.01,0.01\n0.01,0.04,0.01,0.01,0.01\n0.01,0.01,0.04,0.01,0.01\n0.01,0.01,0.01,0.04,0.01\n0.01,0.01,0.01,0.01,0.04')
const method = ref('mean_variance')
const maxWeight = ref(0.3)
const longOnly = ref(true)
const targetReturn = ref('')
const result = ref(null)
const applying = ref(false)
const estimating = ref(false)

async function run() {
  let mu, cov, codes
  try {
    mu = muText.value.split(',').map(Number)
    cov = covText.value.trim().split('\n').map(r => r.split(',').map(Number))
    codes = codesText.value.split(',').map(s => s.trim()).filter(Boolean)
  } catch (e) { toast('mu/cov 格式错误'); return }
  if (mu.some(v => !isFinite(v)) || cov.some(r => r.some(v => !isFinite(v)))) {
    toast('mu/cov 含非数字（NaN/Inf）'); return
  }
  if (cov.length !== mu.length || cov.some(r => r.length !== mu.length)) {
    toast('cov 维度与 mu 不匹配'); return
  }
  if (codes.length && codes.length !== mu.length) {
    toast('codes 数量需与 mu 一致'); return
  }
  try {
    result.value = await api.post('/portfolio-opt', {
      codes, mu, cov, method: method.value, maxWeight: Number(maxWeight.value),
      longOnly: longOnly.value,
      targetReturn: targetReturn.value === '' ? null : Number(targetReturn.value),
    })
    toast(`优化完成 (${result.value.method})`)
  } catch (e) { toast(e.message) }
}

async function applyToPortfolio() {
  if (!result.value?.codes?.length) { toast('请先输入股票代码并求解'); return }
  applying.value = true
  try {
    const res = await api.post('/portfolio-opt/apply', {
      codes: result.value.codes, weights: result.value.weights,
    })
    const ok = res.applied.filter(a => a.qty).length
    const fail = res.applied.length - ok
    toast(`建仓完成：成功 ${ok} 只${fail ? `，失败 ${fail}` : ''}`)
    // 旧版只 toast 不跳转，用户不知道去哪看结果。建仓成功后跳模拟盘页
    if (ok > 0) router.push('/portfolio')
  } catch (e) { toast(e.message) }
  finally { applying.value = false }
}

async function estimateMuCov() {
  const codes = codesText.value.split(',').map(s => s.trim()).filter(Boolean)
  if (codes.length < 2) { toast('请至少输入 2 个股票代码'); return }
  estimating.value = true
  try {
    const res = await api.post('/portfolio-opt/estimate', { codes })
    codesText.value = res.codes.join(',')
    muText.value = res.mu.map(v => v.toFixed(4)).join(',')
    covText.value = res.cov.map(r => r.map(v => v.toFixed(4)).join(',')).join('\n')
    toast(`已从 ${res.codes.length} 只股票历史生成 μ/Σ`)
  } catch (e) { toast(e.message) }
  finally { estimating.value = false }
}

const weightOption = computed(() => {
  if (!result.value?.weights) return {}
  const w = result.value.weights
  const codes = result.value.codes || []
  return {
    tooltip: { trigger: 'item', formatter: '{b}: {d}%' },
    series: [{
      type: 'pie', radius: ['40%', '70%'],
      data: w.map((v, i) => ({ name: codes[i] || `资产${i + 1}`, value: (v * 100).toFixed(2) })),
    }]
  }
})

const fmt = v => v == null ? '-' : Number(v).toFixed(4)
const fmtPct = v => v == null ? '-' : (Number(v) * 100).toFixed(2) + '%'
</script>

<template>
  <div>
    <div class="card">
      <div class="card-head"><h3>组合优化</h3><span class="hint">均值-方差 / 最大Sharpe / 风险平价（cvxpy）</span></div>
      <div class="panel-toolbar">
        <div class="field"><label>方法</label>
          <select v-model="method">
            <option value="mean_variance">均值-方差</option>
            <option value="max_sharpe">最大Sharpe</option>
            <option value="risk_parity">风险平价</option>
            <option value="equal">等权</option>
          </select>
        </div>
        <div class="field"><label>权重上限</label><input v-model="maxWeight" type="number" step="0.05" /></div>
        <div class="field"><label>目标收益(可选)</label><input v-model="targetReturn" type="number" step="0.01" placeholder="0.08" /></div>
        <div class="field"><label>做多</label>
          <select v-model="longOnly"><option :value="true">仅做多</option><option :value="false">允许做空</option></select>
        </div>
      </div>
      <div class="field" style="margin-top:10px"><label>股票代码 codes（逗号分隔，用于落地模拟盘）</label>
        <textarea v-model="codesText" rows="1" style="width:100%;font-family:monospace"></textarea>
      </div>
      <div class="field" style="margin-top:8px"><label>预期收益 mu（逗号分隔）</label>
        <textarea v-model="muText" rows="1" style="width:100%"></textarea>
      </div>
      <div class="field" style="margin-top:8px"><label>协方差矩阵 Σ（每行一行，逗号分隔）</label>
        <textarea v-model="covText" rows="5" style="width:100%;font-family:monospace"></textarea>
      </div>
      <button class="btn-primary" style="margin-top:10px" @click="run">求解</button>
      <button class="btn-ghost" style="margin-top:10px;margin-left:8px" :disabled="estimating" @click="estimateMuCov">{{ estimating ? '生成中…' : '从代码自动生成 μ/Σ' }}</button>
    </div>

    <div v-if="result" class="card">
      <h3>优化结果</h3>
      <div class="kpi-row">
        <div class="kpi"><div class="n">{{ fmtPct(result.stats?.return) }}</div><div class="l">预期收益</div></div>
        <div class="kpi"><div class="n">{{ fmtPct(result.stats?.volatility) }}</div><div class="l">波动率</div></div>
        <div class="kpi"><div class="n">{{ fmt(result.stats?.sharpe) }}</div><div class="l">Sharpe</div></div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:center">
        <div>
          <table class="data-table">
            <thead><tr><th>代码</th><th>权重</th></tr></thead>
            <tbody>
              <tr v-for="(w, i) in result.weights" :key="i">
                <td>{{ result.codes?.[i] || ('资产' + (i + 1)) }}</td>
                <td class="up">{{ fmtPct(w) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <EChart :option="weightOption" style="height:260px" />
      </div>
      <button class="btn-primary" :disabled="applying" style="margin-top:10px" @click="applyToPortfolio">
        {{ applying ? '建仓中…' : '一键建仓模拟盘' }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.data-table th, .data-table td { border: 1px solid var(--border); padding: 8px 11px; text-align: left; }
.data-table th { background: var(--card-2); color: var(--text-dim); font-weight: 600; }
.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 14px 0; }
.kpi { background: var(--card-2); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }
.kpi .n { font-size: 18px; font-weight: 700; color: var(--accent); }
.kpi .l { color: var(--text-mute); font-size: 12px; margin-top: 4px; }
.hint { color: var(--text-mute); font-size: 12px; font-weight: 400; }
.card-head { display: flex; justify-content: space-between; align-items: baseline; }
</style>
