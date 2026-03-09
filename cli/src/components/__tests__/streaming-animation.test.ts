import { describe, it, expect } from 'bun:test'

// ── Catch-up speed formula (from AssistantMessage) ──────────────────────────

function catchUpSpeed(remaining: number): number {
  return Math.max(2, Math.floor(remaining * 0.3))
}

const LINEAR_DRAIN = 8

// ── Linear drain simulation ──────────────────────────────────────────────────

function simulateDrain(
  startDisplayed: number,
  contentLength: number,
  mode: 'catchup' | 'linear',
  ticks: number
): number {
  let displayed = startDisplayed
  for (let i = 0; i < ticks; i++) {
    if (displayed >= contentLength) break
    const remaining = contentLength - displayed
    const speed = mode === 'linear' ? LINEAR_DRAIN : catchUpSpeed(remaining)
    displayed = Math.min(contentLength, displayed + speed)
  }
  return displayed
}

// ── ThinkingStep fade logic ──────────────────────────────────────────────────

function computeFadeStart(prevLength: number, newLength: number): number {
  if (newLength > prevLength) return prevLength
  return -1
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe('catch-up speed formula', () => {
  it('returns minimum 2 for very small remaining', () => {
    expect(catchUpSpeed(1)).toBe(2)
    expect(catchUpSpeed(5)).toBe(2) // floor(5*0.3) = 1, clamped to 2
    expect(catchUpSpeed(6)).toBe(2) // floor(6*0.3) = 1, clamped to 2
  })

  it('returns proportional speed for larger remaining', () => {
    expect(catchUpSpeed(100)).toBe(30)
    expect(catchUpSpeed(50)).toBe(15)
    expect(catchUpSpeed(20)).toBe(6)
  })

  it('always returns at least 2', () => {
    for (let r = 0; r <= 10; r++) {
      expect(catchUpSpeed(r)).toBeGreaterThanOrEqual(2)
    }
  })
})

describe('linear drain (no-snap)', () => {
  it('advances by exactly LINEAR_DRAIN per tick when far from end', () => {
    const displayed = simulateDrain(0, 1000, 'linear', 1)
    expect(displayed).toBe(LINEAR_DRAIN)
  })

  it('does not overshoot content length', () => {
    const displayed = simulateDrain(95, 100, 'linear', 1)
    expect(displayed).toBe(100)
  })

  it('does not snap to end in one tick when many chars remain', () => {
    // With 500 chars remaining and LINEAR_DRAIN=8, one tick only advances 8
    const displayed = simulateDrain(0, 500, 'linear', 1)
    expect(displayed).toBe(LINEAR_DRAIN)
    expect(displayed).toBeLessThan(500)
  })

  it('reaches end after enough ticks', () => {
    const displayed = simulateDrain(0, 80, 'linear', 10) // 10 ticks * 8 = 80
    expect(displayed).toBe(80)
  })
})

describe('catch-up drain', () => {
  it('covers large gaps faster than linear initially', () => {
    const catchup1tick = simulateDrain(0, 1000, 'catchup', 1)
    const linear1tick = simulateDrain(0, 1000, 'linear', 1)
    expect(catchup1tick).toBeGreaterThan(linear1tick)
  })

  it('converges to content length', () => {
    const displayed = simulateDrain(0, 100, 'catchup', 50)
    expect(displayed).toBe(100)
  })
})

// ── ThinkingStep sliding fade window ────────────────────────────────────────

const THINKING_FADE_WINDOW = 15

function computeFadeWindow(displayedLength: number): { fadeWindowStart: number } {
  return { fadeWindowStart: Math.max(0, displayedLength - THINKING_FADE_WINDOW) }
}

describe('ThinkingStep fade logic (sliding window)', () => {
  it('fading portion is last FADE_WINDOW chars of displayed', () => {
    const { fadeWindowStart } = computeFadeWindow(50)
    expect(fadeWindowStart).toBe(35) // 50 - 15
  })

  it('fading portion covers all displayed when displayedLength <= FADE_WINDOW', () => {
    const { fadeWindowStart } = computeFadeWindow(10)
    expect(fadeWindowStart).toBe(0) // max(0, 10-15) = 0
  })

  it('settled portion grows as displayedLength advances', () => {
    const a = computeFadeWindow(20)
    const b = computeFadeWindow(30)
    expect(b.fadeWindowStart).toBeGreaterThan(a.fadeWindowStart)
  })

  it('no fading when displayedLength is 0', () => {
    const { fadeWindowStart } = computeFadeWindow(0)
    expect(fadeWindowStart).toBe(0)
  })
})

describe('ThinkingStep catch-up speed (same as prose)', () => {
  it('uses same formula as AssistantMessage', () => {
    expect(catchUpSpeed(100)).toBe(30)
    expect(catchUpSpeed(50)).toBe(15)
    expect(catchUpSpeed(6)).toBe(2)
  })

  it('linear drain at 8 chars/tick after active stops', () => {
    const displayed = simulateDrain(0, 500, 'linear', 1)
    expect(displayed).toBe(8)
  })
})