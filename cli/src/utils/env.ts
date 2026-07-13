/**
 * CLI environment helper for dependency injection.
 *
 * This module provides CLI-specific env helpers that extend the base
 * process env with CLI-specific vars for terminal/IDE detection.
 */

import type { BaseEnv, CliEnv } from '../types/env'

/**
 * Get base environment values (OS-level vars only).
 */
const readBaseProcessEnv = (): BaseEnv => ({
  HOME: process.env.HOME,
  USERPROFILE: process.env.USERPROFILE,
  APPDATA: process.env.APPDATA,
  XDG_CONFIG_HOME: process.env.XDG_CONFIG_HOME,

  SHELL: process.env.SHELL,
  COMSPEC: process.env.COMSPEC,
  PATH: process.env.PATH,
  NODE_ENV: process.env.NODE_ENV,
  NODE_PATH: process.env.NODE_PATH,

  TERM: process.env.TERM,
  TERM_PROGRAM: process.env.TERM_PROGRAM,
  TERM_BACKGROUND: process.env.TERM_BACKGROUND,
  COLORFGBG: process.env.COLORFGBG,
})

/**
 * Get CLI environment values.
 * Composes from readBaseProcessEnv() + CLI-specific vars.
 */
export const collectCliEnv = (): CliEnv => ({
  ...readBaseProcessEnv(),

  // Editor preferences
  EDITOR: process.env.EDITOR,
  VISUAL: process.env.VISUAL,

  // Multiplexer / terminal session markers
  TMUX: process.env.TMUX,
  STY: process.env.STY,

  // Remote session markers
  SSH_CONNECTION: process.env.SSH_CONNECTION,
  SSH_CLIENT: process.env.SSH_CLIENT,
  SSH_TTY: process.env.SSH_TTY,

  // Terminal/runtime detection
  COLORTERM: process.env.COLORTERM,
  KITTY_WINDOW_ID: process.env.KITTY_WINDOW_ID,
  SIXEL_SUPPORT: process.env.SIXEL_SUPPORT,
  ZED_NODE_ENV: process.env.ZED_NODE_ENV,
  ZED_SHELL: process.env.ZED_SHELL,
  ZED_TERM: process.env.ZED_TERM,

  // Preserve explicit TERM assignment in CLI env payload
  TERM: process.env.TERM,
})

/**
 * Get the raw system process.env object.
 * Use this when you need to pass the full environment to subprocesses
 * or when you need to set environment variables at runtime.
 */
export const getRawProcessEnv = (): NodeJS.ProcessEnv => process.env