import type { MockTurnResponse } from './turn-script'
import { YIELD_USER, YIELD_INVOKE } from '@magnitudedev/xml-act'

type ToolParams = Record<string, string>

function xmlParameter(name: string, value: string): string {
  return `<parameter name="${name}">${value}</parameter>`
}

function xmlInvoke(tool: string, params: ToolParams, body?: string): string {
  const paramLines = Object.entries(params)
    .map(([k, v]) => xmlParameter(k, v))
    .join('\n')
  
  if (body) {
    const bodyParam = xmlParameter('message', body)
    return `<invoke tool="${tool}">\n${paramLines}\n${bodyParam}\n</invoke>`
  }
  
  if (paramLines) {
    return `<invoke tool="${tool}">\n${paramLines}\n</invoke>`
  }
  
  return `<invoke tool="${tool}"/>`
}

function xmlMessage(recipient: string, text: string): string {
  return `<message to="${recipient}">${text}</message>`
}

export class ResponseBuilder {
  private readonly messages: string[] = []
  private readonly tools: string[] = []

  message(text: string): this {
    this.messages.push(xmlMessage('user', text))
    return this
  }

  messageTo(recipient: string, text: string): this {
    this.messages.push(xmlMessage(recipient, text))
    return this
  }

  spawnWorker(id: string, role: string, message: string): this {
    this.tools.push(xmlInvoke('spawn_worker', { id, role }, message))
    return this
  }

  createTask(id: string, type: string, title: string, parent?: string): this {
    const params: ToolParams = { id, type, title }
    if (parent) params.parent = parent
    this.tools.push(xmlInvoke('create_task', params))
    return this
  }

  updateTask(id: string, status: string): this {
    this.tools.push(xmlInvoke('update_task', { id, status }))
    return this
  }

  killWorker(id: string): this {
    this.tools.push(xmlInvoke('kill_worker', { id }))
    return this
  }

  tool(tag: string, params: ToolParams = {}): this {
    this.tools.push(xmlInvoke(tag, params))
    return this
  }

  createAgent(agentId: string, type: string, title: string, message: string): this {
    return this.tool('agent_create', { id: agentId, type, title, message })
  }

  writeArtifact(id: string, content: string): this {
    return this.tool('artifact-write', { id, content })
  }

  updateArtifact(id: string, old: string, new_: string): this {
    return this.tool('artifact-update', { id, old, new: new_ })
  }

  readFile(path: string): this {
    return this.tool('read', { path })
  }

  writeFile(path: string, content: string): this {
    return this.tool('write', { path, content })
  }

  shell(command: string): this {
    return this.tool('shell', { command })
  }

  edit(path: string, old: string, new_: string): this {
    return this.tool('edit', { path, old, new: new_ })
  }

  private build(yieldTag: string): MockTurnResponse {
    const parts: string[] = []
    parts.push(...this.messages)
    parts.push(...this.tools)
    parts.push(yieldTag)
    return { xml: parts.join('') }
  }

  yield(): MockTurnResponse {
    return this.build(YIELD_USER)
  }

  yieldInvoke(): MockTurnResponse {
    return this.build(YIELD_INVOKE)
  }

  next(): MockTurnResponse {
    return this.build(YIELD_USER)
  }
}

export function response(): ResponseBuilder {
  return new ResponseBuilder()
}
