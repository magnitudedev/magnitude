import type { ForkActivityMessage } from '@magnitudedev/agent'

export function sumForkToolCounts(toolCounts: ForkActivityMessage['toolCounts']): number {
  return Object.values(toolCounts).reduce((sum, n) => sum + n, 0)
}



function summarizeToolCounts(toolCounts: ForkActivityMessage['toolCounts']) {
  const entries = Object.entries(toolCounts) as Array<[string, number]>
  const buckets = {
    read: 0,
    edit: 0,
    write: 0,
    search: 0,
    web: 0,
    bash: 0,
    cmd: 0,
    mcp: 0,
  }

  for (const [rawName, count] of entries) {
    if (count <= 0) continue
    const name = rawName.toLowerCase()
    if (name.includes('read')) buckets.read += count
    else if (name.includes('edit') || name.includes('patch') || name.includes('replace')) buckets.edit += count
    else if (name.includes('write') || name.includes('create')) buckets.write += count
    else if (name.includes('search') || name.includes('grep') || name.includes('glob') || name.includes('find')) buckets.search += count
    else if (name.includes('web') || name.includes('fetch') || name.includes('browser')) buckets.web += count
    else if (name.includes('bash') || name.includes('shell') || name.includes('exec')) buckets.bash += count
    else if (name.includes('command') || name.includes('cmd')) buckets.cmd += count
    else if (name.includes('mcp')) buckets.mcp += count
    else buckets.cmd += count
  }

  const plural = (count: number, singular: string, many: string) => `${count} ${count === 1 ? singular : many}`

  return [
    buckets.read > 0 ? `Read ${plural(buckets.read, 'file', 'files')}` : null,
    buckets.write > 0 ? `Wrote ${plural(buckets.write, 'file', 'files')}` : null,
    buckets.edit > 0 ? `Edited ${plural(buckets.edit, 'file', 'files')}` : null,
    buckets.search > 0 ? `Searched ${plural(buckets.search, 'file', 'files')}` : null,
    buckets.web > 0 ? `Ran ${plural(buckets.web, 'web search', 'web searches')}` : null,
    buckets.bash > 0 ? `Ran ${plural(buckets.bash, 'bash command', 'bash commands')}` : null,
    buckets.cmd > 0 ? `Ran ${plural(buckets.cmd, 'command', 'commands')}` : null,
    buckets.mcp > 0 ? `Made ${plural(buckets.mcp, 'MCP call', 'MCP calls')}` : null,
  ].filter((token): token is string => token !== null)
}

export function formatSubagentToolSummaryLine(toolCounts: ForkActivityMessage['toolCounts']): string {
  const tokens = summarizeToolCounts(toolCounts)
  if (tokens.length === 0) return 'no tools yet'
  return tokens.join(', ')
}

export function truncateTaskText(text: string, max = 44): string {
  const normalized = text.trim().replace(/\s+/g, ' ')
  if (normalized.length <= max) return normalized
  return normalized.slice(0, Math.max(0, max - 1)) + '…'
}