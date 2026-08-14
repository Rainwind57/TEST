import { createApp } from 'vue'
import { createPinia } from 'pinia'
import './styles/theme.css'
import App from './App.vue'
import router from './router'
import { useAuthStore } from './stores/auth'

const app = createApp(App)
const pinia = createPinia()
app.use(pinia)

// 挂载前恢复 httpOnly Cookie 登录态，保证路由守卫拿到正确状态
const auth = useAuthStore(pinia)
await auth.bootstrap()

app.use(router)
app.mount('#app')
