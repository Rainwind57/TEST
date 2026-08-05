import { defineStore } from 'pinia'
import api from '../api/client'

export const useWatchlistStore = defineStore('watchlist', {
  state: () => ({
    codes: [],
    source: 'tencent',
    quotes: {},          // { code: quoteObj }
    noData: [],          // 已拉取但无行情数据的代码（无效代码/退市/停牌无数据），用于区分"加载中"与"无数据"
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
      // 行情拉取失败不影响添加结果（无效代码/弱网时刷新会抛错，股票本身已入自选）
      try { await this.refreshQuotes() } catch (e) { /* 忽略，noData 仍会标记 */ }
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
    setSource(src) {
      this.source = src
      this.refreshQuotes()
    }
  }
})
