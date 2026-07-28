export function normalizeCode(raw) {
  raw = String(raw).trim().toLowerCase().replace(/\s/g, '')
  if (raw.startsWith('sh') || raw.startsWith('sz')) return raw
  if (!/^\d{6}$/.test(raw)) return null
  if (raw.startsWith('6') || raw.startsWith('5') || raw.startsWith('9')) return 'sh' + raw
  if (raw.startsWith('0') || raw.startsWith('3')) return 'sz' + raw
  return null
}

export function fmtNum(n, d = 2) {
  return (n === undefined || n === null || isNaN(n)) ? '--' : Number(n).toFixed(d)
}
export function fmtVol(v) {
  return v ? (v / 100).toLocaleString(undefined, { maximumFractionDigits: 0 }) : '--'
}
export function fmtAmount(a) {
  return a ? (a / 10000).toLocaleString(undefined, { maximumFractionDigits: 2 }) : '--'
}
export function fmtPct(p, d = 2) {
  return (p === undefined || p === null || isNaN(p)) ? '--' : `${p >= 0 ? '+' : ''}${Number(p).toFixed(d)}%`
}
export function fmtMoney(v, d = 2) {
  return (v === undefined || v === null || isNaN(v)) ? '--' : `¥${Number(v).toLocaleString(undefined, { maximumFractionDigits: d })}`
}
export function trendCls(p, r) {
  if (!p || !r) return 'flat'
  return p > r ? 'up' : p < r ? 'down' : 'flat'
}
export function stripPrefix(code) {
  return code.replace(/^(sh|sz)/, '').toUpperCase()
}
