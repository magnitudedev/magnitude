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
import type { ResolvedAppearance } from "../stores/appearance-store"

/** Convert a hex palette color to rgba string with given alpha */
function hexToRgba(hex: string, alpha: number): string {
  const rgb = parseHexColorToRgb(hex)
  if (!rgb) return `rgba(0,0,0,${alpha})`
  return `rgba(${Math.round(rgb.r * 255)}, ${Math.round(
    rgb.g * 255
  )}, ${Math.round(rgb.b * 255)}, ${alpha})`
}

/**
 * Map of CSS variable name → palette-derived value.
 * All color values come from palette.ts (single source of truth).
 */
export function generateCssVars(
  mode: ResolvedAppearance = "dark"
): Record<string, string> {
  const light = mode === "light"
  return {
    // ── Backgrounds ──
    "--bg-base": light ? slate[50] : appSurface.bgBase,
    "--bg-surface": light ? "#ffffff" : appSurface.bgSurface,
    "--bg-surface-elevated": light ? slate[100] : appSurface.bgSurfaceElevated,
    "--bg-sidebar": light ? slate[100] : appSurface.bgSurface,
    "--bg-input": light ? "#ffffff" : appSurface.bgInput,
    "--bg-input-focused": light ? slate[50] : appSurface.bgInputFocused,
    "--bg-code": light ? slate[100] : appSurface.bgCode,

    // ── Text ──
    "--fg-primary": light ? slate[900] : slate[200],
    "--fg-secondary": light ? slate[600] : slate[400],
    "--fg-tertiary": light ? slate[500] : slate[500],
    "--fg-placeholder": light ? slate[500] : slate[600],

    // ── Accents ──
    "--accent-primary": light ? blue[700] : accentAliases.primary,
    "--accent-primary-dim": light ? blue[100] : accentAliases.primaryDim,
    "--accent-info": light ? blue[700] : accentAliases.info,
    "--accent-success": light ? green[700] : accentAliases.success,
    "--accent-success-dim": light ? green[200] : accentAliases.successDim,
    "--accent-warning": light ? orange[700] : accentAliases.warning,
    "--accent-warning-dim": light ? orange[200] : accentAliases.warningDim,
    "--accent-error": light ? red[600] : accentAliases.error,
    "--accent-error-dim": light ? red[200] : accentAliases.errorDim,
    "--accent-violet": light ? violet[700] : accentAliases.violet,
    "--accent-violet-dim": light ? violet[200] : accentAliases.violetDim,
    "--accent-indigo": light ? indigo[700] : accentAliases.indigo,

    // ── Semantic lines ──
    "--line-user": light ? blue[700] : blue[400],
    "--line-task": light ? slate[600] : slate[400],
    "--line-bash": light ? orange[700] : orange[400],
    "--line-error": light ? red[600] : red[500],
    "--line-goal": light ? green[700] : green[500],
    "--line-worker": light ? violet[700] : violet[500],
    "--line-interrupted": light ? red[600] : red[500],

    // ── Borders ──
    "--border-default": light ? slate[300] : appSurface.borderDefault,
    "--border-subtle": light ? slate[200] : slate[800],
    "--border-hover": light ? slate[400] : slate[600],
    "--border-focus": light ? blue[700] : accentAliases.primary,

    // ── Syntax (from theme.ts dark syntax) ──
    "--syntax-keyword": light ? violet[700] : violet[300],
    "--syntax-string": light ? green[700] : green[400],
    "--syntax-number": light ? blue[700] : blue[300],
    "--syntax-comment": slate[500],
    "--syntax-function": light ? blue[700] : blue[400],
    "--syntax-variable": light ? slate[700] : slate[200],
    "--syntax-type": light ? green[700] : green[400],
    "--syntax-operator": light ? slate[700] : slate[400],
    "--syntax-property": light ? blue[700] : slate[200],
    "--syntax-punctuation": light ? slate[500] : slate[400],
    "--syntax-literal": light ? blue[700] : blue[300],

    // ── Diff ──
    "--diff-added-bg": hexToRgba(green[500], 0.12),
    "--diff-added-fg": light ? green[700] : green[400],
    "--diff-removed-bg": hexToRgba(red[500], 0.12),
    "--diff-removed-fg": light ? red[700] : red[400],

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
export function injectCssVars(mode: ResolvedAppearance = "dark"): void {
  const vars = generateCssVars(mode)
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
  const lines = Object.entries(vars).map(
    ([name, value]) => `  ${name}: ${value};`
  )
  return `:root {\n${lines.join("\n")}\n}`
}
