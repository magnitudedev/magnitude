import { describe, expect, test } from 'bun:test'
import { createLiveIntegrationHarness, shouldRunLiveProviderTests } from '../src/test-harness/live-integration-harness'

const describeLive = shouldRunLiveProviderTests() ? describe : describe.skip
const harness = shouldRunLiveProviderTests() ? await createLiveIntegrationHarness() : null
const targets = harness ? await harness.getLiveTargets() : []

describeLive('live integration: bound model', () => {
  for (const target of targets) {
    const label = `${target.slot}:${target.model.providerId}:${target.model.id}`

    test(`resolver resolves expected connection for ${label}`, async () => {
      const resolved = await harness!.resolveTarget(target)
      expect(resolved.expectedDriver).toBe(target.expectedDriver)
      expect(resolved.model.id).toBe(target.model.id)
      expect(resolved.model.providerId).toBe(target.model.providerId)
      expect(resolved.bound.connection._tag).toBe(target.expectedDriver === 'openai-responses' ? 'Responses' : 'Baml')
    }, 15000)

    test(`bound complete GenerateChatTitle for ${label}`, async () => {
      const resolved = await harness!.resolveTarget(target)
      const result = await harness!.runBoundGenerateChatTitle(resolved.bound)
      if (result !== null) {
        expect(result.title.trim().length).toBeGreaterThan(0)
      }
    }, 15000)

    test(`bound stream CodingAgentChat for ${label}`, async () => {
      const resolved = await harness!.resolveTarget(target)
      const result = await harness!.runBoundCodingAgentChat(resolved.bound)
      expect(result.text.trim().length).toBeGreaterThan(0)
    }, 30000)
  }
})