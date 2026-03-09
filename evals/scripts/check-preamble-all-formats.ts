import { readFileSync, readdirSync } from 'fs'

const dir = 'evals/results/2026-03-01T07-56-21'
const files = readdirSync(dir).filter(f => f.endsWith('.json'))

for (const f of files) {
  const data = JSON.parse(readFileSync(dir + '/' + f, 'utf8'))
  const modelKey = `${data.provider}:${data.model}`

  // Group by format
  const formatStats: Record<string, { total: number; fail: number; applyFail: number; noResponse: number }> = {}

  for (const s of data.scenarios) {
    const slashIdx = s.scenarioId.indexOf('/')
    const variantKey = s.scenarioId.slice(0, slashIdx)
    if (!variantKey.endsWith(':r0')) continue
    const formatId = variantKey.replace(':r0', '')

    if (!formatStats[formatId]) formatStats[formatId] = { total: 0, fail: 0, applyFail: 0, noResponse: 0 }
    formatStats[formatId].total++

    const passed = Object.values(s.checks as Record<string, { passed: boolean }>).every(c => c.passed)
    if (passed) continue
    formatStats[formatId].fail++

    if (!s.rawResponse || s.rawResponse.length === 0) {
      formatStats[formatId].noResponse++
      continue
    }

    const msg = (s.checks as Record<string, { message?: string }>)['edit-correct']?.message || ''
    if (msg.startsWith('Apply failed:')) {
      formatStats[formatId].applyFail++
    }
  }

  console.log(`=== ${modelKey} ===`)
  for (const [fmt, stats] of Object.entries(formatStats).sort((a, b) => a[0].localeCompare(b[0]))) {
    const passRate = ((stats.total - stats.fail) / stats.total * 100).toFixed(0)
    console.log(`  ${fmt.padEnd(24)} ${passRate}% pass | ${stats.fail} fail (${stats.applyFail} apply-fail, ${stats.noResponse} no-response, ${stats.fail - stats.applyFail - stats.noResponse} content-mismatch)`)
  }
  console.log()
}
