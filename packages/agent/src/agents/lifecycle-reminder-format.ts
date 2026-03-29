export function formatAgentIdList(agentIds: readonly string[]): string {
  if (agentIds.length <= 0) return ''
  if (agentIds.length === 1) return agentIds[0] ?? ''
  if (agentIds.length === 2) return `${agentIds[0]} and ${agentIds[1]}`
  return `${agentIds.slice(0, -1).join(', ')}, and ${agentIds[agentIds.length - 1]}`
}
