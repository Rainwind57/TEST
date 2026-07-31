<script setup>
defineProps({
  type: { type: String, default: 'cards' },   // cards | rows | lines
  count: { type: Number, default: 4 },
})
</script>

<template>
  <div class="skeleton-wrap">
    <template v-if="type === 'cards'">
      <div v-for="i in count" :key="i" class="sk-card">
        <div class="sk-line sk-sm"></div>
        <div class="sk-line sk-lg"></div>
      </div>
    </template>
    <template v-else-if="type === 'rows'">
      <div v-for="i in count" :key="i" class="sk-row">
        <div class="sk-line sk-cell" v-for="j in 6" :key="j"></div>
      </div>
    </template>
    <template v-else>
      <div v-for="i in count" :key="i" class="sk-line sk-block"></div>
    </template>
  </div>
</template>

<style scoped>
.skeleton-wrap { display: contents; }
.sk-line {
  background: linear-gradient(90deg, var(--card-2) 25%, var(--border) 50%, var(--card-2) 75%);
  background-size: 200% 100%;
  animation: sk-shimmer 1.4s infinite;
  border-radius: 6px;
}
@keyframes sk-shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
.sk-card {
  display: inline-flex; flex-direction: column; gap: 10px;
  width: 160px; padding: 16px; background: var(--card);
  border: 1px solid var(--border); border-radius: 12px; margin-right: 12px;
}
.sk-sm { width: 50%; height: 12px; }
.sk-lg { width: 80%; height: 24px; }
.sk-row { display: flex; gap: 12px; padding: 12px 16px; border-bottom: 1px solid var(--border); }
.sk-cell { flex: 1; height: 16px; }
.sk-block { width: 100%; height: 18px; margin-bottom: 12px; }
</style>
