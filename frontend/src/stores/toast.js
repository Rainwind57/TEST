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
    // duration 为 0 时永不自动消失；有 action 默认 10s 后消失，可传 persistent:true 阻止
    const dur = opts.duration ?? (opts.action ? 10000 : 1800)
    if (dur > 0 && !opts.persistent) {
      timer = setTimeout(() => { visible.value = false }, dur)
    }
  }
  function dismiss() {
    visible.value = false
    action.value = null
  }
  return { message, visible, action, toast, dismiss }
}
