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

function taskOpen(id: string, attrs?: XmlAttrs): string {
  return openTag('task', { id, ...(attrs ?? {}) })
}

export class ResponseBuilder {
  private readonly messages: string[] = []
  private readonly explicitTaskBlocks: string[] = []
  private readonly implicitTaskActions: string[] = []

  message(text: string): this {
    this.messages.push(element('message', undefined, text))
    return this
  }

  task(id: string, body: string, attrs?: XmlAttrs): this {
    this.explicitTaskBlocks.push(`${taskOpen(id, attrs)}${body}</task>`)
    return this
  }

  taskMessage(taskId: string, text: string): this {
    return this.task(taskId, element('message', undefined, text))
  }

  taskAssign(taskId: string, role: string, instructions: string): this {
    return this.task(taskId, element('assign', { role }, instructions))
  }

  tool(tag: string, attrs?: XmlAttrs, body?: string): this {
    this.implicitTaskActions.push(element(tag, attrs, body))
    return this
  }

  createAgent(agentId: string, type: string, title: string, message: string): this {
    return this.tool(
      'agent-create',
      { agentId },
      `${element('type', undefined, type)}${element('title', undefined, title)}${element('message', undefined, message)}`,
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

  private build(control: 'idle'): MockTurnResponse {
    const parts: string[] = []
    parts.push(...this.messages)
    parts.push(...this.explicitTaskBlocks)

    if (this.implicitTaskActions.length > 0) {
      parts.push(`${taskOpen('harness-task')}${this.implicitTaskActions.join('')}</task>`)
    }

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
