<script setup>
import Sidebar from './components/Sidebar.vue'
import ToastHost from './components/ToastHost.vue'
</script>

<template>
  <Sidebar />
  <main class="main-content">
    <RouterView />
  </main>
  <ToastHost />
</template>

<style scoped>
.main-content {
  margin-left: var(--sidebar-w);
  padding: 28px 32px 60px;
  min-height: 100vh;
}
@media (max-width: 900px) {
  .main-content { margin-left: 0; }
}
</style>
