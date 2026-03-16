import { violet, rose, indigo, green, slate } from './palette'

export const AGENT_COLOR_PALETTES = [
  {
    name: 'violet',
    border: violet[500],
    bg: '#211e30',
    pulse: [violet[300], violet[400], violet[500], violet[600], violet[700], violet[600], violet[500], violet[400], violet[300]],
  },
  {
    name: 'rose',
    border: rose[500],
    bg: '#271e24',
    pulse: [rose[300], rose[400], rose[500], rose[600], rose[700], rose[600], rose[500], rose[400], rose[300]],
  },
  {
    name: 'indigo',
    border: indigo[500],
    bg: '#1c2130',
    pulse: [indigo[300], indigo[400], indigo[500], indigo[600], indigo[700], indigo[600], indigo[500], indigo[400], indigo[300]],
  },
]

export const ORCHESTRATOR_PALETTE = {
  border: slate[400],
  bg: '#222832',
}

// Green pulse for tab indicator
export const GREEN_PULSE = [green[300], green[400], green[500], green[600], green[700], green[600], green[500], green[400], green[300]]

export function getAgentPalette(colorIndex: number) {
  return AGENT_COLOR_PALETTES[colorIndex % AGENT_COLOR_PALETTES.length]!
}