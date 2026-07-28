import axios from 'axios'

const api = axios.create({
  baseURL: 'http://127.0.0.1:8899/api',
  timeout: 15000
})

api.interceptors.response.use(
  res => res.data,
  err => {
    const msg = err?.response?.data?.detail || err.message || '请求失败'
    return Promise.reject(new Error(msg))
  }
)

export default api
