// ============================================================================
// Color Palette — each hex defined exactly once
// ============================================================================

export const blue = {
  50: '#f0f9ff',
  100: '#e0f2fe',
  200: '#bae6fd',
  300: '#7dd3fc',
  400: '#38bdf8',
  500: '#0ea5e9',
  600: '#0284c7',
  700: '#0369a1',
  800: '#075985',
  900: '#0c4a6e',
} as const

export const slate = {
  50: '#f8fafc',
  100: '#f1f5f9',
  150: '#eaeff6',
  200: '#e2e8f0',
  250: '#d7dfe9',
  300: '#cbd5e1',
  400: '#94a3b8',
  450: '#7d8fa4',
  500: '#64748b',
  600: '#475569',
  700: '#334155',
  750: '#293548',
  800: '#1e293b',
  875: '#16191f',
  900: '#0f1826',
  925: '#101217',
} as const

export const green = {
  200: '#b8f5e1',
  300: '#7ee8c7',
  400: '#4dd4a8',
  500: '#2ab88a',
  600: '#1f9670',
  700: '#17785a',
  800: '#106148',
} as const

export const rose = {
  200: '#fad4de',
  300: '#f2b8c6',
  400: '#e494a7',
  500: '#d17088',
  600: '#b5536e',
  700: '#954458',
} as const

export const violet = {
  200: '#ddd6fe',
  300: '#c4b5fd',
  400: '#a78bfa',
  500: '#8b5cf6',
  600: '#7c3aed',
  700: '#6d28d9',
} as const

export const indigo = {
  200: '#bccef9',
  300: '#93b4f8',
  400: '#6b93f2',
  500: '#4573ea',
  600: '#1d4ed8',
  700: '#1e40af',
} as const

export const orange = {
  200: '#fde0b8',
  300: '#f5c890',
  400: '#e8a55c',
  500: '#d48a38',
  600: '#b87224',
  700: '#965a1a',
} as const

export const red = {
  200: '#fcd4d2',
  300: '#f8ada8',
  400: '#e8736e',
  500: '#d4524d',
  600: '#b93a36',
  700: '#982d2a',
  800: '#7d2321',
} as const

// ============================================================================
// App-specific dark surface tokens — used by the web/desktop app.
// Values must point at the palette scale above rather than introducing
// one-off hex colors.
// ============================================================================

export const appSurface = {
  /** App background */
  bgBase: slate[925],
  /** Panels, composer, sidebar item hover */
  bgSurface: slate[875],
  /** Selected sidebar item, menu popover, modal */
  bgSurfaceElevated: slate[800],
  /** Textarea, form inputs */
  bgInput: slate[800],
  /** Input on focus */
  bgInputFocused: slate[750],
  /** Code block background */
  bgCode: slate[900],
  /** Default border color */
  borderDefault: slate[750],
} as const

// Semantic accent aliases — the spec uses some names that differ slightly
// from the raw palette (e.g., --accent-primary maps to blue.500, but the
// spec's §7.2 originally listed #3b82f6 which is a different blue scale).
// We use the palette as single source of truth.
export const accentAliases = {
  /** Primary accent — blue.500 */
  primary: blue[500],
  /** Primary dim — blue.800 */
  primaryDim: blue[800],
  /** Info accent — blue.500 */
  info: blue[500],
  /** Success — green.500 */
  success: green[500],
  /** Success dim — green.600 */
  successDim: green[600],
  /** Warning — orange.500 */
  warning: orange[500],
  /** Warning dim — orange.600 */
  warningDim: orange[600],
  /** Error — red.500 */
  error: red[500],
  /** Error dim — red.600 */
  errorDim: red[600],
  /** Violet — violet.500 */
  violet: violet[500],
  /** Violet dim — violet.700 */
  violetDim: violet[700],
  /** Indigo — indigo.500 */
  indigo: indigo[500],
} as const
