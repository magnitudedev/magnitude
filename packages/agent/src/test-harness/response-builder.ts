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
  private readonly comms: string[] = []
  private readonly actions: string[] = []

  message(to: string, text: string): this {
    this.comms.push(element('message', { to }, text))
    return this
  }

  messageUser(text: string): this {
    return this.message('user', text)
  }

  messageParent(text: string): this {
    return this.message('parent', text)
  }

  tool(tag: string, attrs?: XmlAttrs, body?: string): this {
    this.actions.push(element(tag, attrs, body))
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

  private build(control: 'yield' | 'next'): MockTurnResponse {
    const comms = this.comms.length > 0 ? element('comms', undefined, this.comms.join('')) : ''
    const actions = this.actions.length > 0 ? element('actions', undefined, this.actions.join('')) : ''
    return { xml: `${actions}${comms}<${control}/>` }
  }

  yield(): MockTurnResponse {
    return this.build('yield')
  }

  next(): MockTurnResponse {
    return this.build('next')
  }
}

export function response(): ResponseBuilder {
  return new ResponseBuilder()
}