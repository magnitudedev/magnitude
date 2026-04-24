import { describe, expect, it } from 'vitest'

import { getModelRecommendation } from './model-recommendations'

describe('model recommendations', () => {
  it('recommends glm-5.1 for Magnitude lead only', () => {
    const recommendation = getModelRecommendation('magnitude', 'glm-5.1')
    expect(recommendation).not.toBeNull()
    expect(recommendation?.classes).toEqual(new Set([
      'lead',
    ]))
  })

  it('recommends kimi-k2.6 for Magnitude worker', () => {
    const recommendation = getModelRecommendation('magnitude', 'kimi-k2.6')
    expect(recommendation).not.toBeNull()
    expect(recommendation?.classes).toEqual(new Set([
      'worker',
    ]))
  })

  it('recommends kimi-k2.6 for Moonshot AI lead and worker', () => {
    const recommendation = getModelRecommendation('moonshotai', 'kimi-k2.6')
    expect(recommendation).not.toBeNull()
    expect(recommendation?.classes.has('lead')).toBe(true)
    expect(recommendation?.classes.has('worker')).toBe(true)
  })
})
