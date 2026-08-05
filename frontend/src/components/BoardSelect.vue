<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'

const props = defineProps({
  modelValue: { type: Array, default: () => ['all'] },
  sectors: { type: Array, default: () => [] },
})
const emit = defineEmits(['update:modelValue'])

const boards = [
  { value: 'all', label: '全部A股' },
  { value: 'sh_main', label: '沪市主板' },
  { value: 'sz_main', label: '深市主板' },
  { value: 'gem', label: '创业板' },
  { value: 'star', label: '科创板' },
  { value: 'bse', label: '北交所' },
  { value: 'hs300', label: '沪深300' },
  { value: 'zz500', label: '中证500' },
  { value: 'etf', label: 'ETF基金' },
]

const open = ref(false)
const selected = computed(() => props.modelValue || ['all'])

function toggle(v) {
  let next
  if (selected.value.includes(v)) next = selected.value.filter(x => x !== v)
  else next = [...selected.value, v]
  if (v === 'all') next = ['all']
  else next = next.filter(x => x !== 'all')
  if (!next.length) next = ['all']
  emit('update:modelValue', next)
}

const label = computed(() => {
  const s = selected.value
  if (s.length === 1 && (s[0] === 'all' || !s[0])) return '全部A股'
  const all = [...boards, ...props.sectors.map(x => ({ value: x.value, label: x.label }))]
  const names = s.map(v => (all.find(o => o.value === v) || {}).label || v)
  return names.join(' + ')
})

function onClickOutside(e) {
  if (!e.target.closest('.board-select-root')) open.value = false
}

onMounted(() => document.addEventListener('click', onClickOutside))
onUnmounted(() => document.removeEventListener('click', onClickOutside))
</script>

<template>
  <div class="board-select-root">
    <div class="field"><label>板块</label>
      <div class="bs-trigger" @click="open = !open">
        <span class="bs-label">{{ label }}</span><span class="bs-arr">▾</span>
      </div>
    </div>
    <div v-if="open" class="bs-pop">
      <label class="bs-item" v-for="o in boards" :key="o.value">
        <input type="checkbox" :checked="selected.includes(o.value)" @change="toggle(o.value)" />
        <span>{{ o.label }}</span>
      </label>
      <template v-if="props.sectors.length">
        <div class="bs-divider">行业板块</div>
        <label class="bs-item" v-for="s in props.sectors" :key="s.value">
          <input type="checkbox" :checked="selected.includes(s.value)" @change="toggle(s.value)" />
          <span>{{ s.label }}</span>
        </label>
      </template>
      <div class="bs-hint">可多选组合，如「沪市主板+创业板」</div>
    </div>
  </div>
</template>

<style scoped>
.board-select-root { position: relative; display: inline-flex; }
.bs-trigger { display: flex; align-items: center; cursor: pointer; background: var(--card-2); border: 1px solid var(--border); border-radius: 8px; padding: 6px 10px; gap: 8px; min-width: 140px; font-size: 13px; color: var(--text); }
.bs-arr { font-size: 10px; color: var(--text-mute); }
.bs-pop { position: absolute; z-index: 100; top: calc(100% + 4px); left: 0; background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 8px 6px; max-height: 340px; overflow-y: auto; min-width: 200px; box-shadow: 0 4px 16px rgba(0,0,0,.25); }
.bs-item { display: flex; align-items: center; gap: 8px; padding: 5px 8px; font-size: 13px; color: var(--text); cursor: pointer; border-radius: 6px; }
.bs-item:hover { background: var(--card-2); }
.bs-item input { margin: 0; }
.bs-divider { padding: 6px 8px 2px; font-size: 11px; color: var(--text-mute); letter-spacing: .5px; }
.bs-hint { padding: 6px 8px; font-size: 11px; color: var(--text-mute); border-top: 1px solid var(--border); margin-top: 4px; }
.field { display: flex; flex-direction: column; gap: 4px; }
.field label { font-size: 12px; color: var(--text-dim); font-weight: 600; }
</style>
