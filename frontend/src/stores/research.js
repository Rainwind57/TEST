import { defineStore } from 'pinia'

// 研究结果跨页共享 store（文档 P0#4：缺少"把 X 结果拿去 Y"的跨页共享状态）
// 存最优参数/当前模型/选股结果，各页面写入 + 下游页面读取预填
export const useResearchStore = defineStore('research', {
  state: () => ({
    optimalParams: null,      // 寻优最优参数 {board,factor,poolSize,groups,n,hist,benchmark,...}
    currentModel: null,       // 当前 ML 模型 {id,...}
    currentSelectResult: null, // 选股结果
  }),
  actions: {
    setOptimalParams(p) { this.optimalParams = p },
    setCurrentModel(m) { this.currentModel = m },
    setCurrentSelectResult(r) { this.currentSelectResult = r },
    consumeOptimalParams() {
      const p = this.optimalParams
      this.optimalParams = null
      return p
    },
  },
})
