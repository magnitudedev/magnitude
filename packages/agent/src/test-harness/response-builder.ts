import type { MockTurnResponse } from './turn-script'
import { YIELD_USER, YIELD_INVOKE } from '@magnitudedev/xml-act'

type MactParams = Record<string, string>

function mactParameter(name: string, value: string): string {
  return `<|parameter:${name}>${value}<parameter|>`
}

function mactInvoke(tool: string, params: MactParams, body?: string): string {
  const paramLines = Object.entries(params)
    .map(([k, v]) => mactParameter(k, v))
    .join('\n')
  
  if (body) {
    // Body must be wrapped in a parameter tag — find the body field name
    // Convention: tools with body content use 'message' as the body parameter
    const bodyParam = mactParameter('message', body)
    return `<|invoke:${tool}>\n${paramLines}\n${bodyParam}<invoke|>`
  }
  
  if (paramLines) {
    return `<|invoke:${tool}>\n${paramLines}\n<invoke|>`
  }
  
  return `<|invoke:${tool}>\n<invoke|>`
}

function mactMessage(recipient: string, text: string): string {
  return `<|message:${recipient}>${text}<message|>`
}

export class ResponseBuilder {
  private readonly messages: string[] = []
  private readonly tools: string[] = []

  message(text: string): this {
    this.messages.push(mactMessage('user', text))
    return this
  }

  messageTo(recipient: string, text: string): this {
    this.messages.push(mactMessage(recipient, text))
    return this
  }

  spawnWorker(id: string, role: string, message: string): this {
    this.tools.push(mactInvoke('spawn-worker', { id, role }, message))
    return this
  }

  createTask(id: string, type: string, title: string, parent?: string): this {
    const params: MactParams = { id, type, title }
    if (parent) params.parent = parent
    this.tools.push(mactInvoke('create-task', params))
    return this
  }

  updateTask(id: string, status: string): this {
    this.tools.push(mactInvoke('update-task', { id, status }))
    return this
  }

  killWorker(id: string): this {
    this.tools.push(mactInvoke('kill-worker', { id }))
    return this
  }

  tool(tag: string, params: MactParams = {}): this {
    this.tools.push(mactInvoke(tag, params))
    return this
  }

  createAgent(agentId: string, type: string, title: string, message: string): this {
    return this.tool('agent-create', { id: agentId, type, title, message })
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
    // 'next' is not a standard yield target, default to user
    return this.build(YIELD_USER)
  }
}

export function response(): ResponseBuilder {
  return new ResponseBuilder()
}
