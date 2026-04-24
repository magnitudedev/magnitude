import type { ScalarType } from '../engine/parameter-schema'

export interface InvocationExampleParameter {
  readonly name: string
  readonly type: ScalarType | 'json_object' | 'json_array'
  readonly required: boolean
}

export interface GenerateInvocationExampleOptions {
  readonly showOptional?: boolean
  readonly compact?: boolean
}

function getPlaceholder(type: ScalarType | 'json_object' | 'json_array'): string {
  if (type === 'string') return '...'
  if (type === 'number') return '123'
  if (type === 'boolean') return 'true'
  if (type === 'json_object') return '{...}'
  if (type === 'json_array') return '[...]'
  return type.values[0] ?? '...'
}

function renderParameter(parameter: InvocationExampleParameter): string {
  const rendered =
    `<magnitude:parameter name="${parameter.name}">` +
    `${getPlaceholder(parameter.type)}` +
    `</magnitude:parameter>`

  return parameter.required ? rendered : `${rendered} <!-- optional -->`
}

export function generateInvocationExample(
  tagName: string,
  parameters: ReadonlyMap<string, InvocationExampleParameter>,
  options: GenerateInvocationExampleOptions = {},
): string {
  const { showOptional = true, compact = false } = options

  const ordered = [...parameters.values()]
    .filter(parameter => showOptional || parameter.required)
    .sort((a, b) => {
      if (a.required === b.required) return 0
      return a.required ? -1 : 1
    })

  if (ordered.length === 0) {
    return `<magnitude:invoke tool="${tagName}"/>`
  }

  const lines = [
    `<magnitude:invoke tool="${tagName}">`,
    ...ordered.map(renderParameter),
    `</magnitude:invoke>`,
  ]

  return compact ? lines.join('') : lines.join('\n')
}
