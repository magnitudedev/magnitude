const ROLE_EMOJI: Record<string, string> = {
  explorer: '❖',
  planner: '⚙',
  builder: '⚒',
  reviewer: '✔',
  debugger: '⛏',
  browser: '⌨',
}

export function getSubagentRoleEmoji(role?: string): string | null {
  if (!role) return null
  return ROLE_EMOJI[role] ?? null
}

export function formatSubagentIdWithEmoji(agentId: string, role?: string): string {
  if (!role) return agentId
  const emoji = getSubagentRoleEmoji(role)
  return emoji ? `${emoji} [${role}] ${agentId}` : `[${role}] ${agentId}`
}
