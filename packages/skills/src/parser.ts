import { remark } from 'remark'
import remarkParse from 'remark-parse'
import type { Root, Content, Html } from 'mdast'
import type { Criteria, Hooks, Phase, SubmitBlock, SubmitField, WorkflowSkill } from './types'

interface FrontmatterResult {
  readonly name: string
  readonly description: string
  readonly body: string
}

interface NodeSlice {
  readonly start: number
  readonly end: number
}

function extractFrontmatter(content: string): FrontmatterResult {
  const match = content.match(/^---\s*\n([\s\S]*?)\n---\s*\n?([\s\S]*)$/)
  if (!match) {
    return { name: '', description: '', body: content }
  }

  const yaml = match[1]
  const body = match[2]
  const data: Record<string, string> = {}

  for (const line of yaml.split('\n')) {
    const idx = line.indexOf(':')
    if (idx === -1) continue
    const key = line.slice(0, idx).trim()
    const rawValue = line.slice(idx + 1).trim()
    const value = rawValue.replace(/^['"]|['"]$/g, '')
    data[key] = value
  }

  return {
    name: data.name ?? '',
    description: data.description ?? '',
    body,
  }
}

function getSlice(node: Content): NodeSlice | undefined {
  const start = node.position?.start.offset
  const end = node.position?.end.offset
  if (typeof start !== 'number' || typeof end !== 'number') return undefined
  return { start, end }
}

function parseAttributes(input: string): Record<string, string> {
  const attrs: Record<string, string> = {}
  const re = /([a-zA-Z0-9_-]+)\s*=\s*"([^"]*)"/g
  let match: RegExpExecArray | null
  while ((match = re.exec(input)) !== null) {
    attrs[match[1]] = match[2]
  }
  return attrs
}

function getTagContent(input: string, tag: string): string | undefined {
  const re = new RegExp(`<${tag}>([\\s\\S]*?)<\\/${tag}>`)
  const match = input.match(re)
  return match?.[1]
}

function parseSubmit(block: string): SubmitBlock | undefined {
  const submitContent = getTagContent(block, 'submit')
  if (!submitContent) return undefined

  const fields: SubmitField[] = []
  const fieldRe = /<(text|file)\b([^>]*?)(?:\/>|>([\s\S]*?)<\/\1>)/g
  let match: RegExpExecArray | null

  while ((match = fieldRe.exec(submitContent)) !== null) {
    const kind = match[1] as 'text' | 'file'
    const attrs = parseAttributes(match[2] ?? '')
    const innerDescription = (match[3] ?? '').trim()
    const description = attrs.description ?? innerDescription

    if (kind === 'text') {
      fields.push({
        type: 'text',
        name: attrs.name ?? '',
        description,
      })
    } else {
      fields.push({
        type: 'file',
        name: attrs.name ?? '',
        fileType: attrs.type,
        description,
      })
    }
  }

  return fields.length > 0 ? { fields } : undefined
}

function parseCriteria(block: string): readonly Criteria[] | undefined {
  const criteriaContent = getTagContent(block, 'criteria')
  if (!criteriaContent) return undefined

  const criteria: Criteria[] = []

  {
    const shellRe = /<shell-succeed\b([^>]*)>([\s\S]*?)<\/shell-succeed>/g
    let match: RegExpExecArray | null
    while ((match = shellRe.exec(criteriaContent)) !== null) {
      const attrs = parseAttributes(match[1] ?? '')
      criteria.push({ type: 'shell-succeed', name: attrs.name ?? '', command: (match[2] ?? '').trim() })
    }
  }

  {
    const userRe = /<user-approval\b([^>]*)>([\s\S]*?)<\/user-approval>/g
    let match: RegExpExecArray | null
    while ((match = userRe.exec(criteriaContent)) !== null) {
      const attrs = parseAttributes(match[1] ?? '')
      criteria.push({ type: 'user-approval', name: attrs.name ?? '', message: (match[2] ?? '').trim() })
    }
  }

  {
    const agentRe = /<agent-approval\b([^>]*)>([\s\S]*?)<\/agent-approval>/g
    let match: RegExpExecArray | null
    while ((match = agentRe.exec(criteriaContent)) !== null) {
      const attrs = parseAttributes(match[1] ?? '')
      criteria.push({
        type: 'agent-approval',
        name: attrs.name ?? '',
        subagent: attrs.subagent ?? '',
        prompt: (match[2] ?? '').trim(),
      })
    }
  }

  return criteria.length > 0 ? criteria : undefined
}

function parseHooks(block: string): Hooks | undefined {
  const hooksContent = getTagContent(block, 'hooks')
  if (!hooksContent) return undefined

  const onStart = getTagContent(hooksContent, 'on-start')?.trim()
  const onSubmit = getTagContent(hooksContent, 'on-submit')?.trim()
  const onAccept = getTagContent(hooksContent, 'on-accept')?.trim()
  const onReject = getTagContent(hooksContent, 'on-reject')?.trim()

  const hooks: Hooks = { onStart, onSubmit, onAccept, onReject }
  return hooks.onStart || hooks.onSubmit || hooks.onAccept || hooks.onReject ? hooks : undefined
}

function parsePhaseBlock(html: string): Omit<Phase, 'prompt'> {
  const trimmed = html.trim()
  const nameMatch = trimmed.match(/<phase\b[^>]*\bname="([^"]+)"/)
  const name = nameMatch?.[1] ?? ''

  const selfClosing = /<phase\b[^>]*\/>\s*$/.test(trimmed)
  if (selfClosing) {
    return { name }
  }

  const innerMatch = trimmed.match(/<phase\b[^>]*>([\s\S]*?)<\/phase>/)
  const inner = innerMatch?.[1] ?? ''

  const submit = parseSubmit(inner)
  const criteria = parseCriteria(inner)
  const hooks = parseHooks(inner)

  return {
    name,
    ...(submit ? { submit } : {}),
    ...(criteria ? { criteria } : {}),
    ...(hooks ? { hooks } : {}),
  }
}

function isPhaseHtml(node: Content): node is Html {
  return node.type === 'html' && /^\s*<phase\b/.test(node.value)
}

export function parseSkill(content: string): WorkflowSkill {
  const { name, description, body } = extractFrontmatter(content)
  const tree = remark().use(remarkParse).parse(body) as Root

  const phaseNodes = tree.children
    .map((node, index) => ({ node, index, slice: getSlice(node) }))
    .filter((entry): entry is { node: Html; index: number; slice: NodeSlice } => isPhaseHtml(entry.node) && !!entry.slice)

  if (phaseNodes.length === 0) {
    return {
      name,
      description,
      preamble: body.trim(),
      phases: [],
    }
  }

  const firstPhaseStart = phaseNodes[0].slice.start
  const preamble = body.slice(0, firstPhaseStart).trim()

  const phases: Phase[] = phaseNodes.map((current, idx) => {
    const next = phaseNodes[idx + 1]
    const promptStart = current.slice.end
    const promptEnd = next ? next.slice.start : body.length
    const prompt = body.slice(promptStart, promptEnd).trim()

    return {
      ...parsePhaseBlock(current.node.value),
      prompt,
    }
  })

  return {
    name,
    description,
    preamble,
    phases,
  }
}
