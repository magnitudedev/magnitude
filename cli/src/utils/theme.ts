import { collectCliEnv } from './env'

import type { MarkdownPalette } from './markdown-content-renderer'
import type { CliEnv } from '../types/env'
import type {
  ChatTheme,
  MarkdownHeadingLevel,
  ThemeName,
} from '../types/theme-system'

/**
 * Theme Configuration System
 *
 * Provides plugin system and customization support for themes
 */

/**
 * Plugin interface for extending theme system
 * Plugins can modify themes at runtime
 */
export interface ThemeExtension {
  /** Unique plugin name */
  name: string
  /**
   * Apply plugin modifications to a theme
   * @param theme - The base theme
   * @param mode - The detected light/dark mode
   * @returns Partial theme to merge
   */
  apply: (
    theme: ChatTheme,
    mode: ThemeName,
  ) => Partial<ChatTheme>
}

/**
 * Main theme configuration interface
 */
export interface ThemeSettings {
  /** Global color overrides applied to themes */
  customColors?: Partial<ChatTheme>
  /** Registered plugins for theme extensions */
  plugins?: ThemeExtension[]
}

/**
 * Default theme configuration
 */
export const defaultThemeSettings: ThemeSettings = {
  customColors: {},
  plugins: [],
}

/**
 * Active theme configuration
 * Can be modified at runtime for customization
 */
export let activeThemeSettings: ThemeSettings = defaultThemeSettings

/**
 * Update the active theme configuration
 * @param config - New configuration (will be merged with defaults)
 */
export const applyThemeSettings = (config: Partial<ThemeSettings>): void => {
  activeThemeSettings = {
    ...defaultThemeSettings,
    ...config,
    plugins: [...(defaultThemeSettings.plugins ?? []), ...(config.plugins ?? [])],
  }
}

/**
 * Register a theme plugin
 * @param plugin - Plugin to register
 */
export const registerThemeExtension = (plugin: ThemeExtension): void => {
  if (!activeThemeSettings.plugins) {
    activeThemeSettings.plugins = []
  }
  // Check if plugin already registered
  if (activeThemeSettings.plugins.some((p) => p.name === plugin.name)) {
    console.warn(`Theme plugin "${plugin.name}" is already registered`)
    return
  }
  activeThemeSettings.plugins.push(plugin)
}

/**
 * Build a complete theme by applying custom colors and plugins
 * All 'default' color values are resolved to actual colors
 * @param baseTheme - The base theme to start from
 * @param mode - Current theme mode (dark or light)
 * @param customColors - Optional custom color overrides
 * @param plugins - Optional theme plugins to apply
 * @returns Complete theme with all customizations applied
 */
export const layerTheme = (
  baseTheme: ChatTheme,
  customColors?: Partial<ChatTheme>,
  plugins?: ThemeExtension[],
  mode: ThemeName = 'dark',
): ChatTheme => {
  const theme = { ...baseTheme }

  if (customColors) {
    Object.assign(theme, customColors)
  }

  for (const plugin of plugins ?? []) {
    Object.assign(theme, plugin.apply(theme, mode))
  }

  resolveFallbackThemeColors(theme, mode)
  theme.name = mode

  return theme
}

/**
 * Resolve 'default' color values to fallback colors
 * Components should never see 'default' - it's resolved during theme building
 */
function resolveFallbackThemeColors(theme: ChatTheme, mode: ThemeName): void {
  const defaultFallback = mode === 'dark' ? '#ffffff' : '#000000'

  const resolve = (value: string, fallback: string = defaultFallback): string => {
    if (typeof value === 'string') {
      const normalized = value.trim().toLowerCase()
      if (normalized === 'default' || normalized.length === 0) {
        return fallback
      }
      return value
    }
    return fallback
  }

  theme.foreground = resolve(theme.foreground)
  theme.muted = resolve(theme.muted)
  theme.inputFg = resolve(theme.inputFg)
  theme.inputFocusedFg = resolve(theme.inputFocusedFg)
}

/**
 * Check if the terminal supports truecolor (24-bit color).
 * Terminals like macOS Terminal.app only support 256 colors and cannot
 * render hex colors properly - they need ANSI color name fallbacks.
 */
// Cache the truecolor support result since it won't change during runtime
let _truecolorSupport: boolean | null = null

export function terminalSupportsRgb24(env: CliEnv = collectCliEnv()): boolean {
  if (_truecolorSupport !== null) {
    return _truecolorSupport
  }

  const termProgram = env.TERM_PROGRAM?.toLowerCase() ?? ''
  const colorterm = env.COLORTERM?.toLowerCase()
  const term = env.TERM?.toLowerCase() ?? ''

  if (termProgram === 'apple_terminal') {
    _truecolorSupport = false
    return false
  }

  if (colorterm === 'truecolor' || colorterm === '24bit') {
    _truecolorSupport = true
    return true
  }

  const truecolorTerminals = [
    'iterm.app',
    'hyper',
    'wezterm',
    'alacritty',
    'kitty',
    'ghostty',
    'vscode',
  ]

  if (truecolorTerminals.some(t => termProgram.includes(t))) {
    _truecolorSupport = true
    return true
  }

  if (term.includes('truecolor') || term.includes('24bit')) {
    _truecolorSupport = true
    return true
  }

  if (term === 'xterm-kitty' || term === 'alacritty' || term.includes('ghostty')) {
    _truecolorSupport = true
    return true
  }

  _truecolorSupport = false
  return false
}

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
  900: '#0f172a',
} as const

export const red = {
  500: '#ef4444',
  600: '#dc2626',
} as const

export const green = {
  500: '#22c55e',
  600: '#059669',
} as const

export const orange = {
  300: '#fdba74',
  400: '#fb923c',
  500: '#f97316',
  600: '#ea580c',
  700: '#c2410c',
  800: '#9a3412',
} as const

export const yellow = {
  400: '#facc15',
} as const

export const amber = {
  400: '#fbbf24',
  500: '#f59e0b',
} as const

export const emerald = {
  400: '#34d399',
  500: '#10b981',
} as const

export const indigo = {
  600: '#1d4ed8',
  700: '#1e40af',
} as const

// ============================================================================
// Theme Definition
// ============================================================================

const DEFAULT_CHAT_THEME: ChatTheme = {
  name: 'dark',
  primary: blue[500],
  secondary: slate[400],
  success: green[500],
  error: red[500],
  warning: orange[600],
  info: blue[500],
  link: blue[400],
  directory: slate[400],

  foreground: slate[100],
  background: 'transparent',
  muted: slate[400],
  border: slate[600],
  surface: slate[900],
  surfaceHover: slate[700],

  aiLine: slate[500],
  userLine: blue[400],
  userMessageBg: slate[800],
  userMessageHoverBg: slate[750],
  inputBg: '#232f41',

  agentToggleHeaderBg: orange[500],
  agentToggleExpandedBg: indigo[600],
  agentFocusedBg: slate[700],
  agentContentBg: '#000000',
  terminalBg: 'transparent',
  inputFg: slate[100],
  inputFocusedFg: '#ffffff',

  modeDefault: blue[500],
  modePlan: amber[400],
  modeAutoBuild: emerald[400],

  imageCardBorder: slate[500],

  markdown: {
    codeBackground: 'transparent',
    codeBorderColor: slate[400],
    codeHeaderFg: slate[500],
    inlineCodeFg: orange[400],
    codeTextFg: slate[100],
    headingFg: {
      1: blue[400],
      2: blue[400],
      3: blue[400],
      4: blue[400],
      5: blue[400],
      6: blue[400],
    },
    listBulletFg: slate[400],
    blockquoteBorderFg: slate[700],
    blockquoteTextFg: slate[200],
    dividerFg: slate[800],
    codeMonochrome: false,
  },
}

const DEFAULT_LIGHT_THEME: ChatTheme = {
  name: 'light',
  primary: blue[600],
  secondary: slate[500],
  success: green[600],
  error: red[500],
  warning: orange[700],
  info: blue[600],
  link: blue[400],
  directory: slate[500],

  foreground: slate[900],
  background: 'transparent',
  muted: slate[500],
  border: slate[300],
  surface: slate[100],
  surfaceHover: slate[200],

  aiLine: slate[500],
  userLine: blue[700],
  userMessageBg: slate[150],
  userMessageHoverBg: slate[200],
  inputBg: slate[150],

  agentToggleHeaderBg: orange[600],
  agentToggleExpandedBg: indigo[600],
  agentFocusedBg: slate[100],
  agentContentBg: '#ffffff',
  terminalBg: '#f0f0f0',
  inputFg: slate[900],
  inputFocusedFg: '#000000',

  modeDefault: blue[600],
  modePlan: amber[500],
  modeAutoBuild: emerald[500],

  imageCardBorder: slate[500],

  markdown: {
    codeBackground: 'transparent',
    codeBorderColor: slate[300],
    codeHeaderFg: slate[500],
    inlineCodeFg: orange[800],
    codeTextFg: slate[900],
    headingFg: {
      1: blue[600],
      2: blue[600],
      3: blue[600],
      4: blue[600],
      5: blue[600],
      6: blue[600],
    },
    listBulletFg: slate[500],
    blockquoteBorderFg: slate[300],
    blockquoteTextFg: slate[700],
    dividerFg: slate[200],
    codeMonochrome: false,
  },
}

export const chatThemes = {
  dark: DEFAULT_CHAT_THEME,
  light: DEFAULT_LIGHT_THEME,
}

// ============================================================================
// Markdown Palette
// ============================================================================

export const buildMarkdownColorPalette = (theme: ChatTheme): MarkdownPalette => {
  const headingDefaults: Record<MarkdownHeadingLevel, string> = {
    1: theme.primary,
    2: theme.primary,
    3: theme.primary,
    4: theme.primary,
    5: theme.primary,
    6: theme.primary,
  }

  const overrides = theme.markdown?.headingFg ?? {}

  const syntaxColors = theme.name === 'light'
    ? {
        keyword: blue[600],
        string: green[600],
        number: blue[600],
        comment: slate[500],
        function: blue[600],
        variable: orange[700],
        type: blue[600],
        operator: blue[600],
        property: blue[600],
        punctuation: slate[500],
        literal: blue[600],
        default: slate[900],
      }
    : {
        keyword: blue[400],
        string: green[500],
        number: blue[400],
        comment: slate[400],
        function: yellow[400],
        variable: orange[400],
        type: blue[400],
        operator: blue[400],
        property: blue[400],
        punctuation: slate[400],
        literal: blue[400],
        default: slate[100],
      }

  return {
    inlineCodeFg: theme.markdown?.inlineCodeFg ?? theme.foreground,
    codeBackground: theme.markdown?.codeBackground ?? 'transparent',
    codeBorderColor: theme.markdown?.codeBorderColor ?? theme.border,
    codeHeaderFg: theme.markdown?.codeHeaderFg ?? theme.muted,
    headingFg: {
      ...headingDefaults,
      ...overrides,
    },
    listBulletFg: theme.markdown?.listBulletFg ?? theme.secondary,
    blockquoteBorderFg: theme.markdown?.blockquoteBorderFg ?? theme.secondary,
    blockquoteTextFg: theme.markdown?.blockquoteTextFg ?? theme.foreground,
    dividerFg: theme.markdown?.dividerFg ?? theme.secondary,
    codeTextFg: theme.markdown?.codeTextFg ?? theme.foreground,
    codeMonochrome: theme.markdown?.codeMonochrome ?? false,
    linkFg: theme.markdown?.linkFg ?? theme.link,
    syntax: syntaxColors,
  }
}

// ============================================================================
// Background Luminance Detection
// ============================================================================

/**
 * Parse a hex color string to RGB components.
 * Supports #RGB, #RRGGBB, and #RRRRGGGGBBBB (16-bit per channel from OSC).
 */
const parseHexToRgb = (hex: string): { r: number; g: number; b: number } | null => {
  const cleaned = hex.replace('#', '')
  if (cleaned.length === 3) {
    return {
      r: parseInt(cleaned[0] + cleaned[0], 16),
      g: parseInt(cleaned[1] + cleaned[1], 16),
      b: parseInt(cleaned[2] + cleaned[2], 16),
    }
  }
  if (cleaned.length === 6) {
    return {
      r: parseInt(cleaned.slice(0, 2), 16),
      g: parseInt(cleaned.slice(2, 4), 16),
      b: parseInt(cleaned.slice(4, 6), 16),
    }
  }
  if (cleaned.length === 12) {
    // 16-bit per channel (e.g., from OSC responses) — take high byte
    return {
      r: parseInt(cleaned.slice(0, 2), 16),
      g: parseInt(cleaned.slice(4, 6), 16),
      b: parseInt(cleaned.slice(8, 10), 16),
    }
  }
  return null
}

/**
 * Check if a hex color represents a light background.
 * Uses ITU-R BT.601 luminance: 0.299*R + 0.587*G + 0.114*B
 * Returns true if luminance > 128 (light), false otherwise (dark).
 */
export const isLightBackground = (hex: string): boolean => {
  const rgb = parseHexToRgb(hex)
  if (!rgb) return false
  const luminance = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b
  return luminance > 128
}

// ============================================================================
// Theme Utilities
// ============================================================================

/**
 * Clone a ChatTheme object to avoid mutations
 */
export const duplicateChatTheme = (input: ChatTheme): ChatTheme => ({
  ...input,
  markdown: input.markdown
    ? {
        ...input.markdown,
        headingFg: input.markdown.headingFg
          ? { ...input.markdown.headingFg }
          : undefined,
      }
    : undefined,
})