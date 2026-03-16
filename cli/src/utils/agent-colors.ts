import { violet, rose, indigo, green, slate, orange, red } from './palette'

export interface AgentColorPalette {
  name: string
  border: string
  bg: string
  pulse: string[]
}

function buildPulse(shades: Record<number, string>): string[] {
  return [shades[300]!, shades[400]!, shades[500]!, shades[600]!, shades[700]!, shades[600]!, shades[500]!, shades[400]!, shades[300]!]
}

export const AGENT_ROLE_PALETTES: Record<string, AgentColorPalette> = {
  orchestrator: { name: 'slate',  border: slate[400],   bg: '#222832', pulse: buildPulse(slate)   },
  explorer:     { name: 'violet', border: violet[500],  bg: '#211e30', pulse: buildPulse(violet)  },
  builder:      { name: 'rose',   border: rose[500],    bg: '#271e24', pulse: buildPulse(rose)    },
  planner:      { name: 'indigo', border: indigo[500],  bg: '#1c2130', pulse: buildPulse(indigo)  },
  reviewer:     { name: 'orange', border: orange[500],  bg: '#2a2018', pulse: buildPulse(orange)  },
  debugger:     { name: 'red',    border: red[500],     bg: '#2a1a1a', pulse: buildPulse(red)     },
  browser:      { name: 'green',  border: green[500],   bg: '#1a2a1e', pulse: buildPulse(green)   },
}

export function getAgentColorByRole(role: string): AgentColorPalette {
  return AGENT_ROLE_PALETTES[role.toLowerCase()] ?? AGENT_ROLE_PALETTES['explorer']!
}

// Legacy — kept for callers not yet migrated
export const AGENT_COLOR_PALETTES = [
  {
    name: 'violet',
    border: violet[500],
    bg: '#211e30',
    pulse: buildPulse(violet),
  },
  {
    name: 'rose',
    border: rose[500],
    bg: '#271e24',
    pulse: buildPulse(rose),
  },
  {
    name: 'indigo',
    border: indigo[500],
    bg: '#1c2130',
    pulse: buildPulse(indigo),
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