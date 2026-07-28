import { defineStore } from 'pinia'
import api from '../api/client'

export const useWatchlistStore = defineStore('watchlist', {
  state: () => ({
    codes: [],
    source: 'tencent',
    quotes: {},          // { code: quoteObj }
    activeCode: null,
    loading: false,
    lastUpdated: null
  }),
  getters: {
    sortedCodes: (state) => state.codes
  },
  actions: {
    async fetchWatchlist() {
      this.codes = await api.get('/watchlist')
      if (!this.activeCode && this.codes.length) this.activeCode = this.codes[0]
    },
    async addCode(code) {
      await api.post('/watchlist', { code })
      await this.fetchWatchlist()
      this.activeCode = code
      await this.refreshQuotes()
    },
    async removeCode(code) {
      await api.delete(`/watchlist/${code}`)
      await this.fetchWatchlist()
      if (this.activeCode === code) this.activeCode = this.codes[0] || null
    },
    async refreshQuotes() {
      if (!this.codes.length) return
      this.loading = true
      try {
        const data = await api.get('/quote', { params: { codes: this.codes.join(','), source: this.source } })
        this.quotes = { ...this.quotes, ...data }
        this.lastUpdated = new Date()
      } finally {
        this.loading = false
      }
    },
    setSource(src) {
      this.source = src
      this.refreshQuotes()
    }
  }
})
