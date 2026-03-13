import { describe, expect, test } from 'bun:test'
import { createLiveIntegrationHarness, shouldRunLiveProviderTests } from '../src/test-harness/live-integration-harness'

const describeLive = shouldRunLiveProviderTests() ? describe : describe.skip
const harness = shouldRunLiveProviderTests() ? await createLiveIntegrationHarness() : null
const targets = harness ? await harness.getLiveTargets() : []

describeLive('live integration: driver', () => {
  for (const target of targets) {
    const label = `${target.slot}:${target.model.providerId}:${target.model.id}`

    test(`driver complete GenerateChatTitle for ${label}`, async () => {
      const result = await harness!.runDriverGenerateChatTitle(target)
      expect(result.collectorData._tag).toBe(target.expectedDriver === 'openai-responses' ? 'Responses' : 'Baml')
      expect(result.usage).toBeTruthy()

      if (target.expectedDriver === 'baml') {
        expect(result.collectorData.rawRequestBody).toBeDefined()
        expect(result.collectorData.rawResponseBody).toBeDefined()
      } else {
        expect(result.collectorData.rawResponseBody).toBeDefined()
      }

      if (result.result !== null) {
        expect(result.result.title.trim().length).toBeGreaterThan(0)
      }
    }, 15000)

    test(`driver stream CodingAgentChat for ${label}`, async () => {
      const result = await harness!.runDriverCodingAgentChat(target)
      expect(result.collectorData._tag).toBe(target.expectedDriver === 'openai-responses' ? 'Responses' : 'Baml')
      expect(result.text.trim().length).toBeGreaterThan(0)
      expect(result.usage).toBeTruthy()

      if (target.expectedDriver === 'baml') {
        expect(result.collectorData.rawRequestBody).toBeDefined()
        expect(result.collectorData.rawResponseBody).toBeDefined()
      } else {
        expect(result.collectorData.rawResponseBody).toBeDefined()
      }
    }, 30000)
  }
})