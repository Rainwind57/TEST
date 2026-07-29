import axios from 'axios'

const baseURL = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8899/api'

const api = axios.create({
  baseURL,
  timeout: 60000
})

api.interceptors.response.use(
  res => res.data,
  err => {
    const msg = err?.response?.data?.detail || err.message || '请求失败'
    return Promise.reject(new Error(msg))
  }
)

export async function downloadFile(url, payload, filename) {
  const res = await axios.post(baseURL + url, payload, {
    responseType: 'blob',
    timeout: 60000
  })
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
