export interface BrowserActionVisual {
  readonly icon: string
  readonly label: string
  readonly detail?: string
}

const BROWSER_ICONS: Record<string, string> = {
  click: '◎',
  doubleClick: '◎◎',
  rightClick: '◎',
  type: '⌨',
  scroll: '↕',
  drag: '⤳',
  navigate: '→',
  goBack: '←',
  switchTab: '⇥',
  newTab: '+',
  screenshot: '◻',
  evaluate: '▶',
}

const BROWSER_BASE_LABELS: Record<string, string> = {
  click: 'Click',
  doubleClick: 'Double-click',
  rightClick: 'Right-click',
  type: 'Type',
  scroll: 'Scroll',
  drag: 'Drag',
  navigate: 'Navigate',
  goBack: 'Go back',
  switchTab: 'Switch tab',
  newTab: 'New tab',
  screenshot: 'Screenshot',
  evaluate: 'Evaluate',
}

function asRecord(input: unknown): Record<string, unknown> {
  return input && typeof input === 'object' ? (input as Record<string, unknown>) : {}
}

function asString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim().length > 0 ? value.trim() : undefined
}

function asNumber(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim().length > 0) {
    const parsed = Number(value.trim())
    return Number.isFinite(parsed) ? parsed : undefined
  }
  return undefined
}

function truncate(value: string, max: number): string {
  return value.length > max ? `${value.slice(0, max - 3)}...` : value
}

function formatInputDetail(toolKey: string, input: unknown): string | undefined {
  const fields = asRecord(input)
  switch (toolKey) {
    case 'click':
    case 'doubleClick':
    case 'rightClick': {
      const x = asNumber(fields.x)
      const y = asNumber(fields.y)
      if (x !== undefined && y !== undefined) return `(${x}, ${y})`
      return undefined
    }
    case 'scroll': {
      const x = asNumber(fields.x)
      const y = asNumber(fields.y)
      const deltaX = asNumber(fields.deltaX)
      const deltaY = asNumber(fields.deltaY)
      const at = x !== undefined && y !== undefined ? `at (${x}, ${y})` : undefined
      const deltaParts: string[] = []
      if (deltaX !== undefined) deltaParts.push(`dx=${deltaX}`)
      if (deltaY !== undefined) deltaParts.push(`dy=${deltaY}`)
      const delta = deltaParts.length > 0 ? deltaParts.join(', ') : undefined
      if (at && delta) return `${at} ${delta}`
      return at ?? delta
    }
    case 'drag': {
      const x1 = asNumber(fields.x1)
      const y1 = asNumber(fields.y1)
      const x2 = asNumber(fields.x2)
      const y2 = asNumber(fields.y2)
      if (x1 !== undefined && y1 !== undefined && x2 !== undefined && y2 !== undefined) {
        return `(${x1}, ${y1}) → (${x2}, ${y2})`
      }
      return undefined
    }
    case 'navigate': {
      const url = asString(fields.url)
      return url ? truncate(url, 80) : undefined
    }
    case 'type': {
      const content = asString(fields.content)
      return content ? `"${truncate(content, 60)}"` : undefined
    }
    case 'switchTab': {
      const index = asNumber(fields.index)
      return index !== undefined ? `#${index}` : undefined
    }
    case 'evaluate': {
      const code = asString(fields.code)
      return code ? truncate(code.replace(/\s+/g, ' '), 60) : undefined
    }
    default:
      return undefined
  }
}

function formatStreamingDetail(toolKey: string, fields: Record<string, unknown>, body?: string): string | undefined {
  const fromFields = formatInputDetail(toolKey, fields)
  if (fromFields) return fromFields
  if (!body || body.trim().length === 0) return undefined
  if (toolKey === 'type') return `"${truncate(body.trim(), 60)}"`
  if (toolKey === 'evaluate') return truncate(body.replace(/\s+/g, ' ').trim(), 60)
  return truncate(body.trim(), 80)
}

export function getBrowserActionIcon(toolKey: string): string {
  return BROWSER_ICONS[toolKey] ?? '◎'
}

export function getBrowserActionBaseLabel(toolKey: string): string {
  return BROWSER_BASE_LABELS[toolKey] ?? 'Browser action'
}

export function formatBrowserActionVisual(toolKey: string, input: unknown): BrowserActionVisual {
  const label = getBrowserActionBaseLabel(toolKey)
  const detail = formatInputDetail(toolKey, input)
  return {
    icon: getBrowserActionIcon(toolKey),
    label,
    ...(detail ? { detail } : {}),
  }
}

export function formatBrowserActionVisualFromStreaming(
  toolKey: string,
  fields: Record<string, unknown>,
  body?: string,
): BrowserActionVisual {
  const label = getBrowserActionBaseLabel(toolKey)
  const detail = formatStreamingDetail(toolKey, fields, body)
  return {
    icon: getBrowserActionIcon(toolKey),
    label,
    ...(detail ? { detail } : {}),
  }
}
