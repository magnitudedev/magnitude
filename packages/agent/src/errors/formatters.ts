export function describeThrown(value: unknown): string {
  if (typeof value === 'string') return value
  if (value instanceof Error) {
    const name = value.name || 'Error'
    return value.message ? `${name}: ${value.message}` : name
  }
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

export function stackTraceLines(value: unknown): readonly string[] {
  if (!(value instanceof Error) || !value.stack) return []
  const frames = value.stack
    .split('\n')
    .slice(1)
    .map((line) => line.trim())
    .filter(Boolean)
  return frames.length > 0 ? ['stack:', ...frames] : []
}
