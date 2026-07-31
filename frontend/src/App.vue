<script setup>
import { ref } from 'vue'
import Sidebar from './components/Sidebar.vue'
import ToastHost from './components/ToastHost.vue'

const sidebarOpen = ref(false)
</script>

<template>
  <Sidebar :open="sidebarOpen" @close="sidebarOpen = false" />
  <main class="main-content">
    <button class="hamburger" @click="sidebarOpen = true">☰</button>
    <RouterView />
  </main>
  <div v-if="sidebarOpen" class="sidebar-mask" @click="sidebarOpen = false"></div>
  <ToastHost />
</template>

<style scoped>
.main-content {
  margin-left: var(--sidebar-w);
  padding: 28px 32px 60px;
  min-height: 100vh;
}
.hamburger {
  display: none;
  position: fixed; top: 12px; left: 12px; z-index: 60;
  width: 40px; height: 40px; border-radius: 8px;
  background: var(--card); border: 1px solid var(--border);
  color: var(--text); font-size: 18px; cursor: pointer;
}
.sidebar-mask {
  display: none;
  position: fixed; inset: 0; z-index: 50;
  background: rgba(0,0,0,.45);
}
@media (max-width: 900px) {
  .main-content { margin-left: 0; padding: 60px 16px 40px; }
  .hamburger { display: block; }
  .sidebar-mask { display: block; }
}
</style>
