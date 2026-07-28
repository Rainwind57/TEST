import { defineStore } from 'pinia'
import api from '../api/client'

export const usePortfolioStore = defineStore('portfolio', {
  state: () => ({
    cash: 0,
    marketValue: 0,
    totalAssets: 0,
    totalPnl: 0,
    totalPnlPct: 0,
    positions: [],
    trades: [],
    equity: [],
    loading: false
  }),
  actions: {
    async fetch() {
      this.loading = true
      try {
        const data = await api.get('/portfolio')
        Object.assign(this, data)
      } finally {
        this.loading = false
      }
    },
    async order(code, side, qty) {
      const data = await api.post('/portfolio/order', { code, side, qty })
      Object.assign(this, data)
    },
    async reset() {
      const data = await api.post('/portfolio/reset')
      Object.assign(this, data)
    }
  }
})
