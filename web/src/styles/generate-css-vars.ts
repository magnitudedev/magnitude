/**
 * Generate CSS custom properties from palette.ts — the single source of truth.
 *
 * Spec §7.2 defines the semantic CSS variable names. This module maps each
 * one to a palette value from `@magnitudedev/client-common`. No hex values
 * are hardcoded here — they all come from palette.ts.
 *
 * At app startup, call `injectCssVars()` to set these as custom properties
 * on `:root`.
 */
import {
  blue,
  slate,
  green,
  violet,
  indigo,
  orange,
  red,
  appSurface,
  accentAliases,
  parseHexColorToRgb,
} from "@magnitudedev/client-common"

/** Convert a hex palette color to rgba string with given alpha */
function hexToRgba(hex: string, alpha: number): string {
  const rgb = parseHexColorToRgb(hex)
  if (!rgb) return `rgba(0,0,0,${alpha})`
  return `rgba(${Math.round(rgb.r * 255)}, ${Math.round(rgb.g * 255)}, ${Math.round(rgb.b * 255)}, ${alpha})`
}

/**
 * Map of CSS variable name → palette-derived value.
 * All color values come from palette.ts (single source of truth).
 */
export function generateCssVars(): Record<string, string> {
  return {
    // ── Backgrounds ──
    "--bg-base": appSurface.bgBase,
    "--bg-surface": appSurface.bgSurface,
    "--bg-surface-elevated": appSurface.bgSurfaceElevated,
    "--bg-sidebar": appSurface.bgSurface,
    "--bg-input": appSurface.bgInput,
    "--bg-input-focused": appSurface.bgInputFocused,
    "--bg-code": appSurface.bgCode,

    // ── Text ──
    "--fg-primary": slate[200],
    "--fg-secondary": slate[400],
    "--fg-tertiary": slate[500],
    "--fg-placeholder": slate[600],

    // ── Accents ──
    "--accent-primary": accentAliases.primary,
    "--accent-primary-dim": accentAliases.primaryDim,
    "--accent-info": accentAliases.info,
    "--accent-success": accentAliases.success,
    "--accent-success-dim": accentAliases.successDim,
    "--accent-warning": accentAliases.warning,
    "--accent-warning-dim": accentAliases.warningDim,
    "--accent-error": accentAliases.error,
    "--accent-error-dim": accentAliases.errorDim,
    "--accent-violet": accentAliases.violet,
    "--accent-violet-dim": accentAliases.violetDim,
    "--accent-indigo": accentAliases.indigo,

    // ── Semantic lines ──
    "--line-user": blue[400],
    "--line-task": slate[400],
    "--line-bash": orange[400],
    "--line-error": red[500],
    "--line-goal": green[500],
    "--line-worker": violet[500],
    "--line-interrupted": red[500],

    // ── Borders ──
    "--border-default": appSurface.borderDefault,
    "--border-subtle": slate[800],
    "--border-hover": slate[600],
    "--border-focus": accentAliases.primary,

    // ── Syntax (from theme.ts dark syntax) ──
    "--syntax-keyword": violet[300],
    "--syntax-string": green[400],
    "--syntax-number": blue[300],
    "--syntax-comment": slate[500],
    "--syntax-function": blue[400],
    "--syntax-variable": slate[200],
    "--syntax-type": green[400],
    "--syntax-operator": slate[400],
    "--syntax-property": slate[200],
    "--syntax-punctuation": slate[400],
    "--syntax-literal": blue[300],

    // ── Diff ──
    "--diff-added-bg": hexToRgba(green[500], 0.12),
    "--diff-added-fg": green[400],
    "--diff-removed-bg": hexToRgba(red[500], 0.12),
    "--diff-removed-fg": red[400],

    // ── Surface tints (for hover/active states) ──
    "--tint-error": hexToRgba(red[500], 0.08),
    "--tint-error-hover": hexToRgba(red[500], 0.16),
    "--tint-warning": hexToRgba(orange[500], 0.08),
  }
}

/**
 * Inject generated CSS variables as custom properties on `:root`.
 * Call this once at app startup before mounting React.
 */
export function injectCssVars(): void {
  const vars = generateCssVars()
  const root = document.documentElement
  for (const [name, value] of Object.entries(vars)) {
    root.style.setProperty(name, value)
  }
}

/**
 * Generate a CSS string representation (for SSR or static extraction).
 */
export function generateCssVarsString(): string {
  const vars = generateCssVars()
  const lines = Object.entries(vars).map(([name, value]) => `  ${name}: ${value};`)
  return `:root {\n${lines.join("\n")}\n}`
}
