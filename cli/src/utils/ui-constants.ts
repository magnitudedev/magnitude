import type { BorderCharacters } from '@opentui/core'

const ROUNDED_CORNERS = {
  topLeft: '╭',
  topRight: '╮',
  bottomLeft: '╰',
  bottomRight: '╯',
} as const

const EDGE_STROKES = {
  horizontal: '─',
  vertical: '│',
} as const

const T_JUNCTIONS = {
  leftT: '├',
  rightT: '┤',
  topT: '┬',
  bottomT: '┴',
  cross: '┼',
} as const

export const BOX_CHARS: BorderCharacters = {
  ...EDGE_STROKES,
  ...ROUNDED_CORNERS,
  ...T_JUNCTIONS,
}

export const TAB_BORDER_CHARS: BorderCharacters = {
  ...BOX_CHARS,
}