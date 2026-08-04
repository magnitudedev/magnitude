import { describe, expect, it } from 'vitest'
import { animationPulse, animationStep } from './animation-clock-store'

describe('animation clock helpers', () => {
  it('derives discrete animation steps from explicit durations', () => {
    expect(animationStep(79, 80)).toBe(0)
    expect(animationStep(80, 80)).toBe(1)
    expect(animationStep(399, 80)).toBe(4)
  })

  it('produces a smooth repeating pulse', () => {
    expect(animationPulse(0, 1_200)).toBeCloseTo(0)
    expect(animationPulse(300, 1_200)).toBeCloseTo(0.5)
    expect(animationPulse(600, 1_200)).toBeCloseTo(1)
    expect(animationPulse(1_200, 1_200)).toBeCloseTo(0)
  })

  it('handles invalid durations without non-finite output', () => {
    expect(animationStep(100, 0)).toBe(0)
    expect(animationPulse(100, 0)).toBe(0)
  })
})
