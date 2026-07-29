import { defineStore } from 'pinia'
import api from '../api/client'

const TOKEN_KEY = 'quant_token'
const USER_KEY = 'quant_user'

export const useAuthStore = defineStore('auth', {
  state: () => ({
    token: localStorage.getItem(TOKEN_KEY) || '',
    user: JSON.parse(localStorage.getItem(USER_KEY) || 'null'),
  }),
  getters: {
    isLoggedIn: (s) => !!s.token,
  },
  actions: {
    setAuth(token, user) {
      this.token = token
      this.user = user
      localStorage.setItem(TOKEN_KEY, token)
      localStorage.setItem(USER_KEY, JSON.stringify(user))
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
    logout() {
      this.token = ''
      this.user = null
      localStorage.removeItem(TOKEN_KEY)
      localStorage.removeItem(USER_KEY)
    },
  },
})

export function authHeaders() {
  const token = localStorage.getItem(TOKEN_KEY)
  return token ? { Authorization: `Bearer ${token}` } : {}
}
