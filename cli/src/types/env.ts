/**
 * CLI-specific environment variable types.
 *
 * Extends base types from common with CLI-specific vars for:
 * - Terminal/IDE detection
 * - Editor preferences
 * - Binary build configuration
 */

/**
 * Base runtime environment variables.
 * These are OS-level env vars common across all packages.
 */
export type BaseEnv = {
  // Shell detection
  SHELL?: string
  COMSPEC?: string // Windows command processor

  // Home directory
  HOME?: string
  USERPROFILE?: string // Windows home
  APPDATA?: string // Windows app data
  XDG_CONFIG_HOME?: string // Linux config home

  // Terminal detection
  TERM?: string
  TERM_PROGRAM?: string
  TERM_BACKGROUND?: string
  COLORFGBG?: string

  // Node/runtime
  NODE_ENV?: string
  NODE_PATH?: string
  PATH?: string
}

/**
 * CLI-specific env vars for terminal/IDE detection and editor preferences.
 */
export type CliEnv = BaseEnv & {
  // Terminal detection (for tmux/screen passthrough)
  TERM?: string
  TMUX?: string
  STY?: string

  // SSH/remote session detection
  SSH_CLIENT?: string
  SSH_TTY?: string
  SSH_CONNECTION?: string

  // Terminal-specific
  KITTY_WINDOW_ID?: string
  SIXEL_SUPPORT?: string
  ZED_NODE_ENV?: string
  ZED_TERM?: string
  ZED_SHELL?: string
  COLORTERM?: string

  // Editor preferences
  VISUAL?: string
  EDITOR?: string
}

/**
 * Function type for getting CLI env values.
 */
export type GetCliEnvFn = () => CliEnv
