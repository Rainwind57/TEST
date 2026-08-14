import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    host: '127.0.0.1',
    port: 5899,
    strictPort: true,
    // 开发代理：前端走相对路径 /api，由 Vite 转发到后端，消除硬编码 baseURL
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8899',
        changeOrigin: true,
      },
    },
  },
})
