/**
 * Shared logprob → color mapping.
 * Single continuous function. No thresholds.
 *
 * Hue: blended power curves — p^0.4 exits red fast, p^2.5 reserves green for top
 *   hue = 120 × (0.5 × p^0.4 + 0.5 × p^2.5)
 * Saturation: 90 × (1 − p⁴⁰) — vivid throughout, fades to white near 100%
 * Lightness: 58 + 37 × p⁴⁰ — bright neon, rises to white near 100%
 */

export function logprobToColor(lp: number): string {
  const p = Math.min(Math.exp(lp), 1)

  const hue = 120 * (0.5 * Math.pow(p, 0.4) + 0.5 * Math.pow(p, 2.5))
  const w = Math.pow(p, 2000)
  const sat = 100 * (1 - w)
  const light = 62 + 33 * w

  return `hsl(${hue}, ${sat}%, ${light}%)`
}

export function logprobToBgColor(lp: number): string {
  const p = Math.min(Math.exp(lp), 1)

  const hue = 120 * (0.5 * Math.pow(p, 0.4) + 0.5 * Math.pow(p, 2.5))
  const w = Math.pow(p, 2000)
  const sat = 100 * (1 - w)
  const light = 58 + 34 * w

  return `hsla(${hue}, ${sat}%, ${light}%, 0.85)`
}

export function logprobToPercent(lp: number): string {
  return (Math.min(Math.exp(lp), 1) * 100).toFixed(1) + '%'
}