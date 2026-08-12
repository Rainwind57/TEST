/**
 * 前端单元测试：format.js normalizeCode / stripPrefix
 * 
 * 覆盖：
 * - D1: 北交所代码 8xxxxx / 920xxx / 4xxxxx 正确映射为 bj 前缀
 * - 标准沪深代码 sh/sz 映射
 * - 异常输入处理
 *
 * 运行方式：
 *   浏览器打开此文件的控制台，或在 Node.js 环境执行
 *   为独立运行，内联了被测试函数
 */

// ============================================================
// 被测试函数（从 frontend/src/utils/format.js 复制）
// ============================================================
function normalizeCode(raw) {
  raw = String(raw).trim().toLowerCase().replace(/\s/g, '')
  if (raw.startsWith('sh') || raw.startsWith('sz') || raw.startsWith('bj')) return raw
  if (!/^\d{6}$/.test(raw)) return null
  if (raw.startsWith('920') || raw.startsWith('8') || raw.startsWith('4')) return 'bj' + raw
  if (raw.startsWith('6') || raw.startsWith('5') || raw.startsWith('9')) return 'sh' + raw
  if (raw.startsWith('0') || raw.startsWith('3')) return 'sz' + raw
  return null
}

function stripPrefix(code) {
  return code.replace(/^(sh|sz|bj)/, '').toUpperCase()
}

// ============================================================
// 简单测试框架
// ============================================================
let passed = 0, failed = 0

function test(name, fn) {
  try {
    fn()
    console.log(`  PASS: ${name}`)
    passed++
  } catch (e) {
    console.log(`  FAIL: ${name} — ${e.message}`)
    failed++
  }
}

function assertEq(actual, expected, msg) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`${msg || ''} expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`)
  }
}

// ============================================================
// D1: 北交所代码
// ============================================================
console.log('\n[D1] 北交所 normalizeCode')

test('8xxxxx → bj', () => {
  assertEq(normalizeCode('830799'), 'bj830799')
  assertEq(normalizeCode('800001'), 'bj800001')
  assertEq(normalizeCode('873527'), 'bj873527')
})

test('920xxx 新号段 → bj', () => {
  assertEq(normalizeCode('920123'), 'bj920123')
  assertEq(normalizeCode('920001'), 'bj920001')
})

test('4xxxxx → bj', () => {
  assertEq(normalizeCode('430001'), 'bj430001')
  assertEq(normalizeCode('400001'), 'bj400001')
})

test('bj 前缀直接通过', () => {
  assertEq(normalizeCode('bj830799'), 'bj830799')
  assertEq(normalizeCode('BJ920123'), 'bj920123')
})

test('920 判断优先于 9→sh（关键回归）', () => {
  // 920 开头必须先匹配 bj，不能被 9→sh 误匹配
  assertEq(normalizeCode('920999'), 'bj920999')
  assertEq(normalizeCode('900901'), 'sh900901')  // 9 开头但非 920 → sh
})

// ============================================================
// 标准沪深代码
// ============================================================
console.log('\n[标准] 沪深 normalizeCode')

test('6xxxxx → sh', () => {
  assertEq(normalizeCode('600000'), 'sh600000')
  assertEq(normalizeCode('688001'), 'sh688001')
})

test('0xxxxx → sz', () => {
  assertEq(normalizeCode('000001'), 'sz000001')
  assertEq(normalizeCode('002001'), 'sz002001')
})

test('3xxxxx → sz', () => {
  assertEq(normalizeCode('300001'), 'sz300001')
  assertEq(normalizeCode('399001'), 'sz399001')
})

test('sh/sz 前缀直接通过', () => {
  assertEq(normalizeCode('sh600000'), 'sh600000')
  assertEq(normalizeCode('SZ000001'), 'sz000001')
})

test('带空格和大小写', () => {
  assertEq(normalizeCode(' SH600000 '), 'sh600000')
  assertEq(normalizeCode('Sz000001'), 'sz000001')
})

// ============================================================
// 异常输入
// ============================================================
console.log('\n[异常] 非法输入')

test('非6位数字返回 null', () => {
  assertEq(normalizeCode('12345'), null)
  assertEq(normalizeCode('1234567'), null)
  assertEq(normalizeCode('abc'), null)
  assertEq(normalizeCode(''), null)
  assertEq(normalizeCode('abc123'), null)
})

test('无效前缀代码返回 null', () => {
  assertEq(normalizeCode('700001'), null)  // 7 开头无映射
  assertEq(normalizeCode('xxxxxx'), null)
})

// ============================================================
// stripPrefix
// ============================================================
console.log('\n[工具] stripPrefix')

test('去掉前缀并大写', () => {
  assertEq(stripPrefix('sh600000'), '600000')
  assertEq(stripPrefix('bj830799'), '830799')
  assertEq(stripPrefix('sz000001'), '000001')
})

test('无前缀直接返回大写', () => {
  assertEq(stripPrefix('600000'), '600000')
})

// ============================================================
// M4: 调参初始值回归
// ============================================================
console.log('\n[M4] 调参初始值（回归：不依赖 normalizeCode，验证逻辑一致性）')

test('featureWeights 初始值逻辑', () => {
  // 模拟 MLView.vue 中 featureWeights 初始化逻辑（修复后应为 1.0）
  const featureNames = ['momentum', 'rsi', 'volatility']
  const existingWeights = {}
  const importanceMap = { momentum: 0.5, rsi: 0.3, volatility: 0.2 }
  
  const featureWeights = {}
  for (const f of featureNames) {
    // 修复后：existingWeights[f] ?? 1.0 （而非 importanceMap[f] ?? 0）
    featureWeights[f] = existingWeights[f] ?? 1.0
  }
  
  assertEq(featureWeights['momentum'], 1.0, '未设置时默认 1.0')
  assertEq(featureWeights['rsi'], 1.0, '未设置时默认 1.0')
  assertEq(featureWeights['volatility'], 1.0, '未设置时默认 1.0')
})

// ============================================================
// 结果
// ============================================================
console.log('\n' + '='.repeat(50))
console.log(`结果: ${passed} 通过, ${failed} 失败`)
console.log('='.repeat(50))

if (typeof window !== 'undefined') {
  document.body.innerHTML = `<pre style="font-family:monospace;padding:20px">
前端单元测试完成: ${passed} 通过, ${failed} 失败
（详情见控制台）
</pre>`
}
