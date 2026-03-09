/**
 * XML Response Parser for Orchestrator Dispatch Eval
 *
 * Regex-based extraction of agent-create, propose, submit, and direct tool
 * tags from the orchestrator's raw XML-ACT response.
 *
 * Does NOT use a DOM parser — XML-ACT is not strict XML.
 */

// =============================================================================
// Types
// =============================================================================

export interface ParsedArtifactCreate {
  id: string
  position: number
}

export interface ParsedAgentCreate {
  agentId: string
  type: string
  title: string
  /** Artifact IDs passed as writable to this agent */
  writableArtifactIds: string[]
  /** Character index of the opening tag in the raw response */
  position: number
}

export interface ParsedPropose {
  title: string
  hasCriteria: boolean
  criteriaCount: number
  hasArtifacts: boolean
  artifactCount: number
  position: number
}

export interface ParsedSubmit {
  summary: string
  position: number
}

export interface ParsedFsRead {
  path: string
  /** Tool ref name for inspect (e.g. 'fs-read', 'fs-read~1') */
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

export interface ParsedAgentDismiss {
  agentId: string
}

export interface ParsedInspectRef {
  toolRef: string
}

export interface ParsedOrchestratorResponse {
  artifactCreates: ParsedArtifactCreate[]
  agentCreates: ParsedAgentCreate[]
  proposes: ParsedPropose[]
  submits: ParsedSubmit[]
  /** All agent types created, in order of appearance */
  agentTypesInOrder: string[]
  /** Direct tool tags used by the orchestrator (e.g. 'fs-read', 'shell') */
  directToolUses: string[]
  /** fs-read calls with paths */
  fsReads: ParsedFsRead[]
  /** fs-search calls */
  fsSearches: ParsedFsSearch[]
  /** fs-tree calls */
  fsTrees: ParsedFsTree[]
  /** shell calls */
  shells: ParsedShell[]
  /** fs-edit calls */
  fsEdits: ParsedFsEdit[]
  /** fs-write calls */
  fsWrites: ParsedFsWrite[]
  /** artifact-read calls with IDs */
  artifactReads: ParsedArtifactRead[]
  /** artifact-write calls */
  artifactWrites: ParsedArtifactWrite[]
  /** agent-dismiss calls */
  agentDismisses: ParsedAgentDismiss[]
  /** inspect ref requests */
  inspectRefs: ParsedInspectRef[]
  hasThinkBlock: boolean
  hasUserMessage: boolean
}

// =============================================================================
// Regex helpers
// =============================================================================

function extractChildTag(xml: string, tag: string): string {
  const re = new RegExp(`<${tag}[^>]*>([\\s\\S]*?)</${tag}>`, 'i')
  const m = xml.match(re)
  return m?.[1]?.trim() ?? ''
}

function countChildTags(xml: string, tag: string): number {
  const re = new RegExp(`<${tag}\\b`, 'gi')
  return (xml.match(re) ?? []).length
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

// =============================================================================
// Main parser
// =============================================================================

export function parseOrchestratorResponse(raw: string): ParsedOrchestratorResponse {
  const artifactCreates: ParsedArtifactCreate[] = []
  const agentCreates: ParsedAgentCreate[] = []
  const proposes: ParsedPropose[] = []
  const submits: ParsedSubmit[] = []
  const directToolUses: string[] = []

  // --- artifact-create ---
  const artifactCreateRe = /<artifact-create\b([^>]*)\s*\/?>/gi
  let m: RegExpExecArray | null
  while ((m = artifactCreateRe.exec(raw)) !== null) {
    artifactCreates.push({
      id: extractAttribute(m[1], 'id'),
      position: m.index,
    })
  }

  // --- agent-create ---
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

  // --- propose ---
  const proposeRe = /<propose\b([^>]*)>([\s\S]*?)<\/propose>|<propose\b([^>]*)\s*\/>/gi
  while ((m = proposeRe.exec(raw)) !== null) {
    const attrs = m[1] ?? m[3] ?? ''
    const body = m[2] ?? ''
    const criteriaCount = countChildTags(body, 'criterion')
    const artifactCount = countChildTags(body, 'artifact')
    proposes.push({
      title: extractAttribute(attrs, 'title'),
      hasCriteria: criteriaCount > 0,
      criteriaCount,
      hasArtifacts: artifactCount > 0,
      artifactCount,
      position: m.index,
    })
  }

  // --- submit ---
  const submitRe = /<submit\b[^>]*>([\s\S]*?)<\/submit>/gi
  while ((m = submitRe.exec(raw)) !== null) {
    submits.push({
      summary: m[1]?.trim() ?? '',
      position: m.index,
    })
  }

  // --- direct tool usage detection ---
  const directToolTags = ['fs-read', 'fs-write', 'fs-edit', 'fs-tree', 'fs-search', 'shell', 'edit', 'write', 'read', 'search', 'tree', 'agent-dismiss', 'agent-pause']
  for (const tag of directToolTags) {
    const re = new RegExp(`<${tag}\\b`, 'i')
    if (re.test(raw)) {
      directToolUses.push(tag)
    }
  }

  // --- fs-read calls (both <fs-read> and <read>) ---
  const fsReads: ParsedFsRead[] = []
  const fsReadRe = /<(?:fs-)?read\b[^>]*\bpath\s*=\s*"([^"]*)"[^>]*\/?>/gi
  let fsReadCount = 0
  while ((m = fsReadRe.exec(raw)) !== null) {
    fsReads.push({
      path: m[1],
      refName: fsReadCount === 0 ? 'fs-read' : `fs-read~${fsReadCount}`,
    })
    fsReadCount++
  }

  // --- fs-search calls (both <fs-search> and <search>) ---
  const fsSearches: ParsedFsSearch[] = []
  const fsSearchRe = /<(?:fs-)?search\b([^>]*)\/?>/gi
  let fsSearchCount = 0
  while ((m = fsSearchRe.exec(raw)) !== null) {
    fsSearches.push({
      pattern: extractAttribute(m[1], 'pattern'),
      path: extractAttribute(m[1], 'path') || '.',
      refName: fsSearchCount === 0 ? 'fs-search' : `fs-search~${fsSearchCount}`,
    })
    fsSearchCount++
  }

  // --- fs-tree calls (both <fs-tree> and <tree>) ---
  const fsTrees: ParsedFsTree[] = []
  const fsTreeRe = /<(?:fs-)?tree\b([^>]*)\/?>/gi
  let fsTreeCount = 0
  while ((m = fsTreeRe.exec(raw)) !== null) {
    fsTrees.push({
      path: extractAttribute(m[1], 'path') || '.',
      refName: fsTreeCount === 0 ? 'fs-tree' : `fs-tree~${fsTreeCount}`,
    })
    fsTreeCount++
  }

  // --- shell calls ---
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

  // --- fs-edit calls (both <fs-edit> and <edit>) ---
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

  // --- fs-write calls (both <fs-write> and <write>) ---
  const fsWrites: ParsedFsWrite[] = []
  const fsWriteRe = /<(?:fs-)?write\b[^>]*\bpath\s*=\s*"([^"]*)"[^>]*>([\s\S]*?)<\/(?:fs-)?write>/gi
  while ((m = fsWriteRe.exec(raw)) !== null) {
    fsWrites.push({
      path: m[1],
      content: m[2],
    })
  }

  // --- artifact-read calls ---
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

  // --- artifact-write calls ---
  const artifactWrites: ParsedArtifactWrite[] = []
  const artifactWriteRe = /<artifact-write\b[^>]*\bid\s*=\s*"([^"]*)"[^>]*>([\s\S]*?)<\/artifact-write>/gi
  while ((m = artifactWriteRe.exec(raw)) !== null) {
    artifactWrites.push({
      id: m[1],
      content: m[2].trim(),
    })
  }

  // --- agent-dismiss calls ---
  const agentDismisses: ParsedAgentDismiss[] = []
  const agentDismissRe = /<agent-dismiss\b[^>]*\bagentId\s*=\s*"([^"]*)"[^>]*\/?>/gi
  while ((m = agentDismissRe.exec(raw)) !== null) {
    agentDismisses.push({ agentId: m[1] })
  }

  // --- inspect refs ---
  const inspectRefs: ParsedInspectRef[] = []
  const refRe = /<ref\s+tool\s*=\s*"([^"]*)"[^>]*\/>/gi
  while ((m = refRe.exec(raw)) !== null) {
    inspectRefs.push({ toolRef: m[1] })
  }

  // --- think block ---
  const hasThinkBlock = /<think\b/i.test(raw) && /<\/think>/i.test(raw)

  // --- user message ---
  let hasUserMessage = false
  const thinkEnd = raw.indexOf('</think>')
  const actionsStart = raw.indexOf('<actions>')

  // Prose between </think> and <actions>
  if (thinkEnd !== -1 && actionsStart !== -1 && actionsStart > thinkEnd) {
    const between = raw.slice(thinkEnd + '</think>'.length, actionsStart).trim()
    if (between.length > 0) hasUserMessage = true
  }
  // Prose after </think> with no <actions> block at all
  if (!hasUserMessage && thinkEnd !== -1 && actionsStart === -1) {
    const afterThink = raw.slice(thinkEnd + '</think>'.length).trim()
    if (afterThink.length > 0) hasUserMessage = true
  }
  // <comms> block anywhere
  if (!hasUserMessage && /<comms>/i.test(raw)) {
    hasUserMessage = true
  }
  // <message to="user"> tag
  if (!hasUserMessage && /<message\b[^>]*to\s*=\s*"user"/i.test(raw)) {
    hasUserMessage = true
  }

  // Build ordered agent types
  const sorted = [...agentCreates].sort((a, b) => a.position - b.position)
  const agentTypesInOrder = sorted.map(a => a.type)

  return {
    artifactCreates,
    agentCreates,
    proposes,
    submits,
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
    agentDismisses,
    inspectRefs,
    hasThinkBlock,
    hasUserMessage,
  }
}
