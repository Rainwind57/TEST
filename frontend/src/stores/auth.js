import { defineStore } from 'pinia'
import api, { setAuthToken, clearAuthToken } from '../api/client'

// 登录凭证迁移：token 仅存内存，持久化改由后端 httpOnly Cookie（JS 不可读）。
// XSS 无法再窃取 localStorage 中的长效凭证；页面刷新后由 bootstrap() 调 /auth/me 恢复。
export const useAuthStore = defineStore('auth', {
  state: () => ({
    token: '',
    user: null,
    bootstrapped: false,
  }),
  getters: {
    isLoggedIn: (s) => !!s.token || !!s.user,
  },
  actions: {
    setAuth(token, user) {
      this.token = token || ''
      this.user = user || null
      if (token) setAuthToken(token)
      else clearAuthToken()
    },
    async register(username, password) {
      const res = await api.post('/auth/register', { username, password })
      this.setAuth(res.token, { id: res.id, username: res.username })
      return res
    },
    async login(username, password) {
      const res = await api.post('/auth/login', { username, password })
      this.setAuth(res.token, { id: res.id, username: res.username })
      return res
    },
    // 页面刷新后从 httpOnly Cookie 恢复登录态（主进程挂载前调用）
    async bootstrap() {
      this.bootstrapped = true
      try {
        const me = await api.get('/auth/me')
        this.user = me
        return true
      } catch {
        this.user = null
        return false
      }
    },
    async logout() {
      try { await api.post('/auth/logout') } catch { /* 忽略网络异常 */ }
      this.token = ''
      this.user = null
      clearAuthToken()
    },
  },
})
