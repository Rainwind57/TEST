import { ref } from 'vue'

const message = ref('')
const visible = ref(false)
const action = ref(null)   // { label, onClick } | null；有 action 时不自动消失
let timer = null

export function useToast() {
  function toast(msg, opts = {}) {
    message.value = msg
    action.value = opts.action || null
    visible.value = true
    clearTimeout(timer)
    // 错误类（带 action 或 persistent）不自动消失，需用户手动关闭；普通提示 1.8s
    if (!opts.action && !opts.persistent) {
      timer = setTimeout(() => { visible.value = false }, opts.duration || 1800)
    }
  }
  function dismiss() {
    visible.value = false
    action.value = null
  }
  return { message, visible, action, toast, dismiss }
}
