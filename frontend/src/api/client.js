import axios from 'axios'
import { useToast } from '../stores/toast'

const baseURL = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8899/api'
export { baseURL }

const api = axios.create({
  baseURL,
  timeout: 300000  // 旧版 60s：选股/回测/ML 大池操作经常跑不完被误判超时，放宽到 5 分钟
})

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('quant_token')
  if (token) {
    config.headers = config.headers || {}
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

const { toast } = useToast()

api.interceptors.response.use(
  res => res.data,
  err => {
    const status = err?.response?.status
    if (status === 401) {
      // token 失效/未登录：清除凭证并跳登录页（带 redirect 回跳）
      localStorage.removeItem('quant_token')
      localStorage.removeItem('quant_user')
      const path = window.location.pathname + window.location.search
      if (!path.startsWith('/login')) {
        window.location.href = '/login?redirect=' + encodeURIComponent(path)
      }
      return Promise.reject(new Error('登录已失效，请重新登录'))
    }
    const msg = err?.response?.data?.detail || err.message || '请求失败'
    // 网络错误（无 response）或 5xx：toast 带重试按钮，不自动消失（旧版仅 toast 1.8s 消失无重试）
    const isNetwork = !err.response
    const is5xx = status >= 500
    if (isNetwork || is5xx) {
      const config = err.config
      toast(msg, { action: { label: '重试', onClick: () => api.request(config) } })
      return Promise.reject(new Error(msg))
    }
    return Promise.reject(new Error(msg))
  }
)

// 长任务请求（ML 训练 / 回测 / 寻优），超时放宽到 5 分钟，避免 60s 默认超时误判失败
export async function longTask(url, payload, { timeout = 300000 } = {}) {
  return api.post(url, payload, { timeout })
}

export async function downloadFile(url, payload, filename) {
  const token = localStorage.getItem('quant_token')
  const res = await axios.post(baseURL + url, payload, {
    responseType: 'blob',
    timeout: 300000,
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  })
  saveBlob(res, filename)
}

// GET 下载（报告历史文件等只读端点），响应处理与 downloadFile 一致
export async function downloadGet(url, filename) {
  const token = localStorage.getItem('quant_token')
  const res = await axios.get(baseURL + url, {
    responseType: 'blob',
    timeout: 120000,
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  })
  saveBlob(res, filename)
}

function saveBlob(res, filename) {
  // 检验输入为二进制载荷，输出为文件下载——仅 JSON 错误和纯文本错误需拦截
  const ct = res.headers['content-type'] || ''
  if (ct.includes('application/json') || ct.includes('text/plain')) {
    const txt = res.data.text()
    return txt.then((t) => {
      let msg = '下载失败'
      try { const j = JSON.parse(t); msg = j.detail || j.message || msg }
      catch (e) { msg = (t || '').slice(0, 200) || msg }
      throw new Error(msg)
    })
  }
  const disposition = res.headers['content-disposition'] || ''
  const match = disposition.match(/filename=([^;]+)/)
  const name = filename || (match ? match[1].replace(/"/g, '') : 'download')
  const blob = new Blob([res.data])
  const link = document.createElement('a')
  link.href = URL.createObjectURL(blob)
  link.download = name
  link.click()
  URL.revokeObjectURL(link.href)
}

export default api
