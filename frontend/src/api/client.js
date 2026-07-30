import axios from 'axios'

const baseURL = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8899/api'
export { baseURL }

const api = axios.create({
  baseURL,
  timeout: 60000
})

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('quant_token')
  if (token) {
    config.headers = config.headers || {}
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

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
    }
    const msg = err?.response?.data?.detail || err.message || '请求失败'
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
  // 校验响应类型：后端报错返回 JSON，不应下成损坏文件
  const ct = res.headers['content-type'] || ''
  if (ct.includes('json') || ct.includes('text/')) {
    const txt = await res.data.text()
    let msg = '下载失败'
    try { const j = JSON.parse(txt); msg = j.detail || j.message || msg }
    catch (e) { msg = (txt || '').slice(0, 200) || msg }
    throw new Error(msg)
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
