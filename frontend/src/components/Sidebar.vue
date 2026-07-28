<script setup>
import { RouterLink, useRoute } from 'vue-router'

const route = useRoute()
const items = [
  { path: '/quote', label: '行情', icon: 'Q' },
  { path: '/factor', label: '因子', icon: 'F' },
  { path: '/regression', label: '回归', icon: 'R' },
  { path: '/factor-regression', label: '多因子回归', icon: 'M' },
  { path: '/select', label: '选股', icon: 'S' },
  { path: '/backtest', label: '分层回测', icon: 'B' },
  { path: '/portfolio', label: '模拟盘', icon: 'P' }
]
</script>

<template>
  <aside class="sidebar">
    <div class="brand">
      <div class="logo">Q</div>
      <div>
        <div class="title">量化研究平台</div>
        <div class="sub">免费接口聚合</div>
      </div>
    </div>
    <nav>
      <RouterLink
        v-for="item in items"
        :key="item.path"
        :to="item.path"
        class="nav-item"
        :class="{ active: route.path === item.path }"
      >
        <span class="nav-icon">{{ item.icon }}</span>
        <span>{{ item.label }}</span>
      </RouterLink>
    </nav>
    <div class="sidebar-footer">
      <div class="dot-row"><span class="dot"></span><span>数据源：腾讯 / 新浪 / 东财</span></div>
    </div>
  </aside>
</template>

<style scoped>
.sidebar {
  width: var(--sidebar-w);
  height: 100vh;
  position: fixed;
  top: 0; left: 0;
  background: var(--card);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  padding: 22px 16px;
}
.brand { display: flex; align-items: center; gap: 12px; margin-bottom: 28px; padding: 0 6px; }
.logo {
  width: 36px; height: 36px;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-weight: 800; font-size: 16px; color: white;
  box-shadow: 0 0 16px rgba(79,140,255,.4);
  flex-shrink: 0;
}
.title { font-size: 14px; font-weight: 700; }
.sub { font-size: 11px; color: var(--text-mute); margin-top: 2px; }

nav { display: flex; flex-direction: column; gap: 4px; flex: 1; }
.nav-item {
  display: flex; align-items: center; gap: 12px;
  padding: 11px 14px;
  border-radius: 10px;
  color: var(--text-dim);
  font-size: 13px;
  font-weight: 600;
  transition: all .2s;
}
.nav-item:hover { background: var(--card-2); color: var(--text); }
.nav-item.active {
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  color: white;
  box-shadow: 0 4px 14px rgba(79,140,255,.3);
}
.nav-icon {
  width: 22px; height: 22px;
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 700;
  background: rgba(255,255,255,.08);
  border-radius: 6px;
}

.sidebar-footer { padding: 12px 6px 0; border-top: 1px solid var(--border); }
.dot-row { display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--text-mute); }
.dot { width: 6px; height: 6px; border-radius: 50%; background: var(--down); }
</style>
