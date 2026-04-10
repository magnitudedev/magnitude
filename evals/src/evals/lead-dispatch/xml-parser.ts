/**
 * XML Response Parser for Orchestrator Dispatch Eval
 *
 * Regex-based extraction of agent/direct-tool tags from lead XML-ACT responses.
 * Does NOT use a strict XML parser.
 */

export interface ParsedArtifactCreate {
  id: string
  position: number
}

export interface ParsedAgentCreate {
  agentId: string
  type: string
  title: string
  writableArtifactIds: string[]
  position: number
}

export interface ParsedMessage {
  to: string
  body: string
  position: number
}

export interface ParsedFsRead {
  path: string
  refName: string
}

export interface ParsedFsSearch {
  pattern: string
  path: string
  refName: string
}

export interface ParsedFsTree {
  path: string
  refName: string
}

export interface ParsedShell {
  command: string
  refName: string
}

export interface ParsedFsEdit {
  path: string
  oldText: string
  newText: string
}

export interface ParsedFsWrite {
  path: string
  content: string
}

export interface ParsedArtifactRead {
  id: string
  refName: string
}

export interface ParsedArtifactWrite {
  id: string
  content: string
}

type TurnControl = 'idle' | null

export interface ParsedInspectRef {
  toolRef: string
}

export interface ParsedOrchestratorResponse {
  artifactCreates: ParsedArtifactCreate[]
  agentCreates: ParsedAgentCreate[]
  messages: ParsedMessage[]

  agentTypesInOrder: string[]
  directToolUses: string[]
  fsReads: ParsedFsRead[]
  fsSearches: ParsedFsSearch[]
  fsTrees: ParsedFsTree[]
  shells: ParsedShell[]
  fsEdits: ParsedFsEdit[]
  fsWrites: ParsedFsWrite[]
  artifactReads: ParsedArtifactRead[]
  artifactWrites: ParsedArtifactWrite[]
  inspectRefs: ParsedInspectRef[]
  hasThinkBlock: boolean
  hasUserMessage: boolean
  turnControl: TurnControl
  firstActionKind: string | null
}

function extractChildTag(xml: string, tag: string): string {
  const re = new RegExp(`<${tag}[^>]*>([\\s\\S]*?)</${tag}>`, 'i')
  const m = xml.match(re)
  return m?.[1]?.trim() ?? ''
}

function extractAttribute(tag: string, attr: string): string {
  const re = new RegExp(`\\b${attr}\\s*=\\s*"([^"]*)"`, 'i')
  const m = tag.match(re)
  return m?.[1] ?? ''
}

function extractAllAttributes(xml: string, tag: string, attr: string): string[] {
  const re = new RegExp(`<${tag}\\b[^>]*\\b${attr}\\s*=\\s*"([^"]*)"[^>]*/?>`, 'gi')
  const results: string[] = []
  let match: RegExpExecArray | null
  while ((match = re.exec(xml)) !== null) {
    results.push(match[1])
  }
  return results
}

function detectTurnControl(raw: string): TurnControl {
  return /<idle\s*\/>/i.test(raw) ? 'idle' : null
}

function detectFirstActionKind(raw: string): string | null {
  const actionPatterns: Array<{ re: RegExp; kind: string }> = [
    { re: /<agent-create\b/gi, kind: 'agent:create' },
    { re: /<artifact-create\b/gi, kind: 'artifact:create' },
    { re: /<(?:fs-)?read\b/gi, kind: 'tool:read' },
    { re: /<(?:fs-)?search\b/gi, kind: 'tool:grep' },
    { re: /<(?:fs-)?tree\b/gi, kind: 'tool:tree' },
    { re: /<shell\b/gi, kind: 'tool:shell' },
    { re: /<(?:fs-)?edit\b/gi, kind: 'tool:fs-edit' },
    { re: /<(?:fs-)?write\b/gi, kind: 'tool:write' },
    { re: /<artifact-read\b/gi, kind: 'tool:artifact-read' },
    { re: /<artifact-write\b/gi, kind: 'tool:artifact-write' },
  ]

  let best: { index: number; kind: string } | null = null
  for (const entry of actionPatterns) {
    const m = entry.re.exec(raw)
    if (!m) continue
    if (!best || m.index < best.index) best = { index: m.index, kind: entry.kind }
  }

  return best?.kind ?? null
}

export function parseOrchestratorResponse(raw: string): ParsedOrchestratorResponse {
  const artifactCreates: ParsedArtifactCreate[] = []
  const agentCreates: ParsedAgentCreate[] = []
  const messages: ParsedMessage[] = []

  const directToolUses: string[] = []

  // artifact-create
  const artifactCreateRe = /<artifact-create\b([^>]*)\s*\/?>/gi
  let m: RegExpExecArray | null
  while ((m = artifactCreateRe.exec(raw)) !== null) {
    artifactCreates.push({
      id: extractAttribute(m[1], 'id'),
      position: m.index,
    })
  }

  // agent-create
  const agentCreateRe = /<agent-create\b([^>]*)>([\s\S]*?)<\/agent-create>/gi
  while ((m = agentCreateRe.exec(raw)) !== null) {
    const attrs = m[1]
    const body = m[2]
    agentCreates.push({
      agentId: extractAttribute(attrs, 'agentId'),
      type: extractChildTag(body, 'type'),
      title: extractChildTag(body, 'title'),
      writableArtifactIds: extractAllAttributes(body, 'writable-artifact', 'id'),
      position: m.index,
    })
  }

  // message
  const messageRe = /<message\b([^>]*)>([\s\S]*?)<\/message>/gi
  while ((m = messageRe.exec(raw)) !== null) {
    messages.push({
      to: extractAttribute(m[1], 'to'),
      body: m[2].trim(),
      position: m.index,
    })
  }

  // direct tool usage detection
  const directToolTags = ['read', 'write', 'fs-edit', 'tree', 'grep', 'shell', 'edit', 'write', 'read', 'search', 'tree', 'artifact-read', 'artifact-write']
  for (const tag of directToolTags) {
    const re = new RegExp(`<${tag}\\b`, 'i')
    if (re.test(raw)) directToolUses.push(tag)
  }

  // read (both read + read)
  const fsReads: ParsedFsRead[] = []
  const fsReadRe = /<(?:fs-)?read\b[^>]*\bpath\s*=\s*"([^"]*)"[^>]*\/?>/gi
  let fsReadCount = 0
  while ((m = fsReadRe.exec(raw)) !== null) {
    fsReads.push({
      path: m[1],
      refName: fsReadCount === 0 ? 'read' : `read~${fsReadCount}`,
    })
    fsReadCount++
  }

  // grep (both grep + search)
  const fsSearches: ParsedFsSearch[] = []
  const fsSearchRe = /<(?:fs-)?search\b([^>]*)\/?>/gi
  let fsSearchCount = 0
  while ((m = fsSearchRe.exec(raw)) !== null) {
    fsSearches.push({
      pattern: extractAttribute(m[1], 'pattern'),
      path: extractAttribute(m[1], 'path') || '.',
      refName: fsSearchCount === 0 ? 'grep' : `grep~${fsSearchCount}`,
    })
    fsSearchCount++
  }

  // tree (both tree + tree)
  const fsTrees: ParsedFsTree[] = []
  const fsTreeRe = /<(?:fs-)?tree\b([^>]*)\/?>/gi
  let fsTreeCount = 0
  while ((m = fsTreeRe.exec(raw)) !== null) {
    fsTrees.push({
      path: extractAttribute(m[1], 'path') || '.',
      refName: fsTreeCount === 0 ? 'tree' : `tree~${fsTreeCount}`,
    })
    fsTreeCount++
  }

  // shell
  const shells: ParsedShell[] = []
  const shellRe = /<shell\b[^>]*>([\s\S]*?)<\/shell>/gi
  let shellCount = 0
  while ((m = shellRe.exec(raw)) !== null) {
    shells.push({
      command: m[1].trim(),
      refName: shellCount === 0 ? 'shell' : `shell~${shellCount}`,
    })
    shellCount++
  }

  // fs-edit (both fs-edit + edit)
  const fsEdits: ParsedFsEdit[] = []
  const fsEditRe = /<(?:fs-)?edit\b[^>]*\bpath\s*=\s*"([^"]*)"[^>]*>([\s\S]*?)<\/(?:fs-)?edit>/gi
  while ((m = fsEditRe.exec(raw)) !== null) {
    const body = m[2]
    fsEdits.push({
      path: m[1],
      oldText: extractChildTag(body, 'old'),
      newText: extractChildTag(body, 'new'),
    })
  }

  // write (both write + write)
  const fsWrites: ParsedFsWrite[] = []
  const fsWriteRe = /<(?:fs-)?write\b[^>]*\bpath\s*=\s*"([^"]*)"[^>]*>([\s\S]*?)<\/(?:fs-)?write>/gi
  while ((m = fsWriteRe.exec(raw)) !== null) {
    fsWrites.push({
      path: m[1],
      content: m[2],
    })
  }

  // artifact-read
  const artifactReads: ParsedArtifactRead[] = []
  const artifactReadRe = /<artifact-read\b[^>]*\bid\s*=\s*"([^"]*)"[^>]*\/?>/gi
  let artReadCount = 0
  while ((m = artifactReadRe.exec(raw)) !== null) {
    artifactReads.push({
      id: m[1],
      refName: artReadCount === 0 ? 'artifact-read' : `artifact-read~${artReadCount}`,
    })
    artReadCount++
  }

  // artifact-write
  const artifactWrites: ParsedArtifactWrite[] = []
  const artifactWriteRe = /<artifact-write\b[^>]*\bid\s*=\s*"([^"]*)"[^>]*>([\s\S]*?)<\/artifact-write>/gi
  while ((m = artifactWriteRe.exec(raw)) !== null) {
    artifactWrites.push({
      id: m[1],
      content: m[2].trim(),
    })
  }

  // inspect refs
  const inspectRefs: ParsedInspectRef[] = []
  const refRe = /<ref\s+tool\s*=\s*"([^"]*)"[^>]*\/>/gi
  while ((m = refRe.exec(raw)) !== null) {
    inspectRefs.push({ toolRef: m[1] })
  }

  const hasThinkBlock = /<lenses>/i.test(raw) && /<\/lenses>/i.test(raw)
  const hasMessageToUser = messages.some(msg => msg.to.toLowerCase() === 'user')

  let hasUserMessage = hasMessageToUser
  if (!hasUserMessage && /<comms>/i.test(raw)) hasUserMessage = true

  const thinkEnd = Math.max(raw.indexOf('</think>'), raw.indexOf('</lenses>'))
  const thinkEndLen = raw.indexOf('</lenses>') > raw.indexOf('</think>') ? '</lenses>'.length : '</think>'.length
  const actionsStart = raw.indexOf('<actions>')
  if (!hasUserMessage && thinkEnd !== -1 && actionsStart !== -1 && actionsStart > thinkEnd) {
    const between = raw.slice(thinkEnd + thinkEndLen, actionsStart).trim()
    if (between.length > 0) hasUserMessage = true
  }
  if (!hasUserMessage && thinkEnd !== -1 && actionsStart === -1) {
    const afterThink = raw.slice(thinkEnd + thinkEndLen).trim()
    if (afterThink.length > 0) hasUserMessage = true
  }

  const sorted = [...agentCreates].sort((a, b) => a.position - b.position)
  const agentTypesInOrder = sorted.map(a => a.type)

  return {
    artifactCreates,
    agentCreates,
    messages,
    agentTypesInOrder,
    directToolUses,
    fsReads,
    fsSearches,
    fsTrees,
    shells,
    fsEdits,
    fsWrites,
    artifactReads,
    artifactWrites,
    inspectRefs,
    hasThinkBlock,
    hasUserMessage,
    turnControl: detectTurnControl(raw),
    firstActionKind: detectFirstActionKind(raw),
  }
}