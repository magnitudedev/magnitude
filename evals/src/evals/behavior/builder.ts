import type { Scenario, Check } from '../../types'
import { mockProject } from './mock-project'

const AO = '<actions>'
const AC = '</actions>'

export function makeRef(tool: string, content = ''): string {
  return ['<', 'ref tool="', tool, '">', content, '</', 'ref>'].join('')
}

export interface JudgeCheck {
  id: string
  description: string
  question: string
}

export interface BehaviorScenario extends Scenario {
  judgeChecks?: JudgeCheck[]
}

interface Message {
  role: 'assistant' | 'user'
  content: string[]
}

// ─── Turn Builder ────────────────────────────────────────────────────────────

export interface DeployAgentOpts {
  agentId: string
  type: 'explorer' | 'planner' | 'builder' | 'debugger' | 'reviewer' | 'browser'
  title: string
  prompt: string
}

export interface TurnBuilder {
  think(text: string, about?: string): TurnBuilder
  message(text: string): TurnBuilder
  deployAgent(opts: DeployAgentOpts): TurnBuilder
  readFiles(paths: string | string[], overrides?: Record<string, string>): TurnBuilder
  shell(command: string): TurnBuilder
  fsTree(path: string): TurnBuilder
  fsSearch(pattern: string, path?: string): TurnBuilder
  action(xml: string): TurnBuilder
  _pendingFileReads(): { paths: string[]; overrides: Record<string, string> }
  build(): string
}

class TurnBuilderImpl implements TurnBuilder {
  private thinks: string[] = []
  private messages: string[] = []
  private actions: string[] = []
  private fileReadPaths: string[] = []
  private fileReadOverrides: Record<string, string> = {}

  think(text: string, about: string = 'task'): TurnBuilder {
    this.thinks.push(`<think about="${about}">${text}</think>`)
    return this
  }

  message(text: string): TurnBuilder {
    this.messages.push(text)
    return this
  }

  deployAgent({ agentId, type, title, prompt }: DeployAgentOpts): TurnBuilder {
    this.actions.push(
      `<agent-create agentId="${agentId}">\n` +
      `<type>${type}</type>\n` +
      `<title>${title}</title>\n` +
      `<prompt>${prompt}</prompt>\n` +
      `</agent-create>`
    )
    return this
  }

  readFiles(paths: string | string[], overrides: Record<string, string> = {}): TurnBuilder {
    const pathArr = Array.isArray(paths) ? paths : [paths]
    this.fileReadPaths.push(...pathArr)
    Object.assign(this.fileReadOverrides, overrides)
    for (const path of pathArr) {
      this.actions.push(`<read path="${path}" />`)
    }
    const n = pathArr.length
    const refs: string[] = []
    for (let i = n - 1; i >= 1; i--) refs.push(makeRef(`read~${i}`))
    if (n > 0) refs.push(makeRef('read'))
    this.actions.push(`<inspect>\n${refs.join('\n')}\n</inspect>`)
    return this
  }

  shell(command: string): TurnBuilder {
    this.actions.push(`<shell>${command}</shell>`)
    return this
  }

  fsTree(path: string): TurnBuilder {
    this.actions.push(`<tree path="${path}" />`)
    return this
  }

  fsSearch(pattern: string, path?: string): TurnBuilder {
    this.actions.push(
      path
        ? `<grep path="${path}"><pattern>${pattern}</pattern></grep>`
        : `<grep><pattern>${pattern}</pattern></grep>`
    )
    return this
  }

  action(xml: string): TurnBuilder {
    this.actions.push(xml)
    return this
  }

  _pendingFileReads(): { paths: string[]; overrides: Record<string, string> } {
    return { paths: this.fileReadPaths, overrides: this.fileReadOverrides }
  }

  build(): string {
    const parts: string[] = []
    parts.push(...this.thinks)
    parts.push(...this.messages)
    if (this.actions.length > 0) {
      parts.push(`${AO}\n${this.actions.join('\n')}\n${AC}`)
    }
    return parts.join('\n')
  }
}

// ─── File Result Injection ───────────────────────────────────────────────────

function readFile(path: string, overrides: Record<string, string>): string {
  if (path in overrides) return overrides[path]
  try { return mockProject.read(path) } catch { return `(file not found: ${path})` }
}

function buildFileToolResults(paths: string[], overrides: Record<string, string>): string {
  const n = paths.length
  const refs: string[] = []
  // paths[0] was first read → read~(n-1), paths[n-1] was last → read
  for (let i = n - 1; i >= 1; i--) {
    refs.push(makeRef(`read~${i}`, readFile(paths[n - 1 - i], overrides)))
  }
  refs.push(makeRef('read', readFile(paths[n - 1], overrides)))
  return `<results>\n<inspect>\n${refs.join('\n')}\n</inspect>\n</results>`
}

// ─── Scenario Builder ────────────────────────────────────────────────────────

export interface AgentResponseOpts {
  artifact?: { id: string; type: string; content: string }
  agentStatuses?: Record<string, string>
}

export interface ScenarioBuilder {
  description(text: string): ScenarioBuilder
  context(sessionContextStr: string): ScenarioBuilder
  user(text: string): ScenarioBuilder
  assistant(fn: (t: TurnBuilder) => TurnBuilder): ScenarioBuilder
  agentResponse(agentId: string, message: string, opts?: AgentResponseOpts): ScenarioBuilder
  judge(question: string, id?: string): ScenarioBuilder
  check(id: string, fn: (response: string) => boolean, message?: string): ScenarioBuilder
  build(): BehaviorScenario
}

function userMsg(text: string): string {
  return `<user mode="text" at="2026-Mar-04 10:00:00">\n${text}\n</user>`
}

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, '')
    .trim()
    .split(/\s+/)
    .slice(0, 6)
    .join('-')
}

class ScenarioBuilderImpl implements ScenarioBuilder {
  private readonly id: string
  private desc = ''
  private messages: Message[] = []
  private checks: Check[] = []
  private judgeChecks: JudgeCheck[] = []

  constructor(id: string) {
    this.id = id
  }

  description(text: string): ScenarioBuilder {
    this.desc = text
    return this
  }

  context(sessionContextStr: string): ScenarioBuilder {
    this.messages.push({ role: 'user', content: [sessionContextStr] })
    return this
  }

  user(text: string): ScenarioBuilder {
    this.messages.push({ role: 'user', content: [userMsg(text)] })
    return this
  }

  assistant(fn: (t: TurnBuilder) => TurnBuilder): ScenarioBuilder {
    const t = fn(new TurnBuilderImpl())
    this.messages.push({ role: 'assistant', content: [t.build()] })
    const { paths, overrides } = t._pendingFileReads()
    if (paths.length > 0) {
      this.messages.push({ role: 'user', content: [buildFileToolResults(paths, overrides)] })
    }
    return this
  }

  agentResponse(agentId: string, message: string, opts?: AgentResponseOpts): ScenarioBuilder {
    const chunks = [
      '<results>\n</results>',
      `<agent_response from="${agentId}">\n${message}\n</agent_response>`,
    ]
    if (opts?.artifact) {
      const { id, type, content } = opts.artifact
      chunks.push(`<artifact id="${id}" type="${type}">\n${content}\n</artifact>`)
    }
    if (opts?.agentStatuses) {
      const lines = Object.entries(opts.agentStatuses).map(([id, s]) => `- ${id}: ${s}`)
      chunks.push(`<agents_status>\n${lines.join('\n')}\n</agents_status>`)
    }
    this.messages.push({ role: 'user', content: [chunks.join('\n')] })
    return this
  }

  judge(question: string, id?: string): ScenarioBuilder {
    this.judgeChecks.push({ id: id ?? slugify(question), description: question, question })
    return this
  }

  check(id: string, fn: (response: string) => boolean, message = 'check failed'): ScenarioBuilder {
    this.checks.push({
      id,
      description: id,
      evaluate(rawResponse) {
        const passed = fn(rawResponse)
        return { passed, message: passed ? undefined : message }
      },
    })
    return this
  }

  build(): BehaviorScenario {
    return {
      id: this.id,
      description: this.desc,
      messages: this.messages,
      checks: this.checks,
      judgeChecks: this.judgeChecks,
    }
  }
}

export function scenario(id: string): ScenarioBuilder {
  return new ScenarioBuilderImpl(id)
}