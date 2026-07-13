export function formatTokensCompact(n: number): string {
  if (n >= 1000) {
    const v = (n / 1000).toFixed(1)
    return (v.endsWith('.0') ? v.slice(0, -2) : v) + 'k'
  }
  return `${n}`
}
