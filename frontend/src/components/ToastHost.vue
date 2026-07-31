<script setup>
import { useToast } from '../stores/toast'
const { message, visible, action, dismiss } = useToast()
</script>

<template>
  <div class="toast" :class="{ show: visible, 'toast-error': action }">
    <span>{{ message }}</span>
    <button v-if="action" class="toast-action" @click="action.onClick(); dismiss()">{{ action.label }}</button>
    <button v-if="action" class="toast-close" @click="dismiss">×</button>
  </div>
</template>

<style scoped>
.toast-action {
  margin-left: 12px; padding: 3px 10px; border-radius: 6px;
  background: var(--accent); color: #fff; border: none; cursor: pointer; font-size: 12px;
}
.toast-close {
  margin-left: 8px; background: transparent; border: none; color: var(--text-mute);
  cursor: pointer; font-size: 16px; line-height: 1;
}
</style>
