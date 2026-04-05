import type { MockTurnResponse } from './turn-script'

type XmlAttrs = Record<string, string>

function openTag(tag: string, attrs?: XmlAttrs): string {
  if (!attrs || Object.keys(attrs).length === 0) return `<${tag}>`
  const rendered = Object.entries(attrs)
    .map(([k, v]) => `${k}="${v}"`)
    .join(' ')
  return `<${tag} ${rendered}>`
}

function element(tag: string, attrs?: XmlAttrs, body?: string): string {
  if (body === undefined) {
    if (!attrs || Object.keys(attrs).length === 0) return `<${tag}/>`
    const rendered = Object.entries(attrs)
      .map(([k, v]) => `${k}="${v}"`)
      .join(' ')
    return `<${tag} ${rendered}/>`
  }
  return `${openTag(tag, attrs)}${body}</${tag}>`
}

export class ResponseBuilder {
  private readonly messages: string[] = []
  private readonly tools: string[] = []

  message(text: string): this {
    this.messages.push(element('message', undefined, text))
    return this
  }

  spawnWorker(id: string, role: string): this {
    this.tools.push(element('spawn-worker', { id, role }))
    return this
  }

  createTask(id: string, type: string, title: string, parent?: string): this {
    const attrs: XmlAttrs = { id, type, title }
    if (parent) attrs.parent = parent
    this.tools.push(element('create-task', attrs))
    return this
  }

  updateTask(id: string, status: string): this {
    this.tools.push(element('update-task', { id, status }))
    return this
  }

  killWorker(id: string): this {
    this.tools.push(element('kill-worker', { id }))
    return this
  }

  tool(tag: string, attrs?: XmlAttrs, body?: string): this {
    this.tools.push(element(tag, attrs, body))
    return this
  }

  createAgent(agentId: string, type: string, title: string, message: string): this {
    return this.tool(
      'agent-create',
      { id: agentId, type },
      `${element('title', undefined, title)}${element('message', undefined, message)}`,
    )
  }

  writeArtifact(id: string, content: string): this {
    return this.tool('artifact-write', { id }, content)
  }

  updateArtifact(id: string, old: string, new_: string): this {
    return this.tool(
      'artifact-update',
      { id },
      `${element('old', undefined, old)}${element('new', undefined, new_)}`,
    )
  }

  readFile(path: string): this {
    return this.tool('read', { path })
  }

  writeFile(path: string, content: string): this {
    return this.tool('write', { path }, content)
  }

  shell(command: string): this {
    return this.tool('shell', undefined, command)
  }

  edit(path: string, old: string, new_: string): this {
    return this.tool(
      'edit',
      { path },
      `${element('old', undefined, old)}${element('new', undefined, new_)}`,
    )
  }

  private build(control: string): MockTurnResponse {
    const parts: string[] = []
    parts.push(...this.messages)
    parts.push(...this.tools)
    parts.push(`<${control}/>`)
    return { xml: parts.join('') }
  }

  yield(): MockTurnResponse {
    return this.build('idle')
  }

  next(): MockTurnResponse {
    return this.build('next')
  }
}

export function response(): ResponseBuilder {
  return new ResponseBuilder()
}
