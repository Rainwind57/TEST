<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'
import { useToast } from '../stores/toast'

const { toast } = useToast()
const auth = useAuthStore()
const router = useRouter()

const mode = ref('login')
const username = ref('')
const password = ref('')
const loading = ref(false)

async function submit() {
  if (!username.value || !password.value) { toast('请输入用户名和密码'); return }
  loading.value = true
  try {
    if (mode.value === 'login') {
      await auth.login(username.value, password.value)
    } else {
      await auth.register(username.value, password.value)
    }
    toast(`欢迎，${auth.user.username}`)
    router.push('/quote')
  } catch (e) { toast(e.message) }
  finally { loading.value = false }
}
</script>

<template>
  <div class="login-wrap">
    <div class="login-card">
      <div class="brand">
        <div class="logo">Q</div>
        <div>
          <div class="title">量化研究平台</div>
          <div class="sub">登录后可保存策略、回测档案、自定义因子</div>
        </div>
      </div>
      <div class="tabs">
        <button :class="{ active: mode === 'login' }" @click="mode = 'login'">登录</button>
        <button :class="{ active: mode === 'register' }" @click="mode = 'register'">注册</button>
      </div>
      <div class="field"><label>用户名</label><input v-model="username" @keyup.enter="submit" /></div>
      <div class="field"><label>密码</label><input v-model="password" type="password" @keyup.enter="submit" /></div>
      <button class="btn-primary" :disabled="loading" @click="submit">
        {{ loading ? '处理中…' : (mode === 'login' ? '登录' : '注册') }}
      </button>
      <div class="hint">匿名使用也可，但策略/存档按用户隔离</div>
    </div>
  </div>
</template>

<style scoped>
.login-wrap { display: flex; align-items: center; justify-content: center; min-height: 70vh; }
.login-card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 32px; width: 360px; }
.brand { display: flex; align-items: center; gap: 12px; margin-bottom: 24px; }
.logo { width: 40px; height: 40px; background: linear-gradient(135deg, var(--accent), var(--accent-2)); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-weight: 800; color: white; }
.title { font-size: 16px; font-weight: 700; }
.sub { font-size: 11px; color: var(--text-mute); margin-top: 2px; }
.tabs { display: flex; gap: 8px; margin-bottom: 18px; }
.tabs button { flex: 1; padding: 8px; border: 1px solid var(--border); background: var(--card-2); color: var(--text-dim); border-radius: 8px; cursor: pointer; font-size: 13px; }
.tabs button.active { background: var(--accent); color: white; border-color: var(--accent); }
.field { margin-bottom: 14px; }
.field label { display: block; font-size: 12px; color: var(--text-mute); margin-bottom: 6px; }
.field input { width: 100%; padding: 9px 12px; border: 1px solid var(--border); background: var(--card-2); color: var(--text); border-radius: 8px; font-size: 14px; }
.btn-primary { width: 100%; margin-top: 6px; }
.hint { margin-top: 14px; font-size: 12px; color: var(--text-mute); text-align: center; }
</style>
