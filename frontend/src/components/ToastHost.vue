<script setup>
import { useToast } from '../stores/toast'
const { message, visible } = useToast()
</script>

<template>
  <div class="toast" :class="{ show: visible }">{{ message }}</div>
</template>
