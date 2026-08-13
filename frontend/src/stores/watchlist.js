import { defineStore } from 'pinia'
import api from '../api/client'

export const useWatchlistStore = defineStore('watchlist', {
  state: () => ({
    codes: [],
    nameMap: {},          // { code: name } — 从 /watchlist API 返回
    source: 'tencent',
    quotes: {},
    noData: [],
    activeCode: null,
    loading: false,
    lastUpdated: null
  }),
  getters: {
    sortedCodes: (state) => state.codes,
    getName: (state) => (code) => state.nameMap[code] || state.quotes[code]?.name || code
  },
  actions: {
    async fetchWatchlist() {
      const data = await api.get('/watchlist')
      // 兼容旧格式（字符串数组）和新格式（{code, name} 数组）
      if (data.length && typeof data[0] === 'object') {
        this.codes = data.map(d => d.code)
        this.nameMap = {}
        data.forEach(d => { if (d.name) this.nameMap[d.code] = d.name })
      } else {
        this.codes = data
      }
      if (!this.activeCode && this.codes.length) this.activeCode = this.codes[0]
    },
    async addCode(code, name = '') {
      await api.post('/watchlist', { code, name })
      await this.fetchWatchlist()
      this.activeCode = code
      try { await this.refreshQuotes() } catch (e) { /* ignore */ }
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
        // 拉取完成后：有 codes 但始终无行情数据的即视为"无数据"（而非永久"加载中"）
        this.noData = this.codes.filter(c => !this.quotes[c])
      }
    },
    async fetchQuote(code) {
      const data = await api.get('/quote', { params: { codes: code, source: this.source } })
      return data[code] || null
    },
    setSource(src) {
      this.source = src
      this.refreshQuotes()
    }
  }
})
