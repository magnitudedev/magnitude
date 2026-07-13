export function formatTime(timestamp: number, timezone: string | null): string {
  const d = new Date(timestamp)
  return new Intl.DateTimeFormat('en-GB', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
    timeZone: timezone ?? undefined,
  }).format(d)
}

export function formatDayTime(timestamp: number, timezone: string | null): string {
  const d = new Date(timestamp)
  const date = new Intl.DateTimeFormat('sv-SE', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    timeZone: timezone ?? undefined,
  }).format(d)
  return `${date} ${formatTime(timestamp, timezone)}`
}

export function dateKey(timestamp: number, timezone: string | null): string {
  const d = new Date(timestamp)
  return new Intl.DateTimeFormat('sv-SE', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    timeZone: timezone ?? undefined,
  }).format(d)
}

export function minuteKey(timestamp: number, timezone: string | null): string {
  const d = new Date(timestamp)
  return new Intl.DateTimeFormat('sv-SE', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
    timeZone: timezone ?? undefined,
  }).format(d)
}

export interface TimeBoundaryEmitter {
  next(timestamp: number): string | null
}

export function createTimeBoundaryEmitter(timezone: string | null): TimeBoundaryEmitter {
  let lastTimeBoundaryMinuteKey: string | null = null
  let lastTimeBoundaryDateKey: string | null = null

  return {
    next(timestamp: number): string | null {
      const currentMinute = minuteKey(timestamp, timezone)
      if (currentMinute === lastTimeBoundaryMinuteKey) return null

      const currentDate = dateKey(timestamp, timezone)
      const showDate = lastTimeBoundaryMinuteKey == null || currentDate !== lastTimeBoundaryDateKey
      lastTimeBoundaryMinuteKey = currentMinute
      lastTimeBoundaryDateKey = currentDate

      return `--- ${showDate ? formatDayTime(timestamp, timezone) : formatTime(timestamp, timezone)} ---`
    },
  }
}
