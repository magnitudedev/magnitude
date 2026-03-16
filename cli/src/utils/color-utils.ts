/**
 * Color utilities for agent UI components.
 */

/**
 * Parse a 6-digit hex color to RGB components.
 */
function parseHex(hex: string): { r: number; g: number; b: number } | null {
  const cleaned = hex.replace('#', '')
  if (cleaned.length !== 6) return null
  return {
    r: parseInt(cleaned.slice(0, 2), 16),
    g: parseInt(cleaned.slice(2, 4), 16),
    b: parseInt(cleaned.slice(4, 6), 16),
  }
}

function toHex(r: number, g: number, b: number): string {
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`
}

// slate[900] = #0f172a — used as the dark base for tinted backgrounds
const SURFACE_DARK = { r: 15, g: 23, b: 42 }

/**
 * Return a very dark tinted version of the given hex color,
 * suitable as a subtle background. Mixes the color toward slate[900]
 * so saturated colors (orange, red) still produce readable dark backgrounds.
 */
export function tintColor(hex: string, intensity = 0.14): string {
  const rgb = parseHex(hex)
  if (!rgb) return '#0f172a'
  return toHex(
    Math.round(SURFACE_DARK.r + (rgb.r - SURFACE_DARK.r) * intensity),
    Math.round(SURFACE_DARK.g + (rgb.g - SURFACE_DARK.g) * intensity),
    Math.round(SURFACE_DARK.b + (rgb.b - SURFACE_DARK.b) * intensity),
  )
}

/**
 * Dim a color by blending toward black at the given factor (0–1).
 */
export function dimColor(hex: string, factor = 0.45): string {
  const rgb = parseHex(hex)
  if (!rgb) return '#000000'
  return toHex(
    Math.round(rgb.r * factor),
    Math.round(rgb.g * factor),
    Math.round(rgb.b * factor),
  )
}