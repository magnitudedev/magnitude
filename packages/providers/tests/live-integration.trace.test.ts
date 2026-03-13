import { describe, expect, test } from 'bun:test'
import { createLiveIntegrationHarness, shouldRunLiveProviderTests } from '../src/test-harness/live-integration-harness'

const describeLive = shouldRunLiveProviderTests() ? describe : describe.skip
const harness = shouldRunLiveProviderTests() ? await createLiveIntegrationHarness() : null
const targets = harness ? await harness.getLiveTargets() : []

describeLive('live integration: trace', () => {
  for (const target of targets) {
    const label = `${target.slot}:${target.model.providerId}:${target.model.id}`

    test(`trace emitted for GenerateChatTitle on ${label}`, async () => {
      harness!.traces.clear()
      const resolved = await harness!.resolveTarget(target)
      await harness!.runBoundGenerateChatTitle(resolved.bound)

      expect(harness!.traces.traces.length).toBeGreaterThanOrEqual(1)
      const trace = harness!.traces.traces.at(-1)!
      assertCommon(trace, target)
      expect(trace.request).toBeTruthy()
      expect(trace.request.messages ?? trace.request.input).toBeTruthy()
      expect(trace.response).toBeTruthy()
      expect(trace.response.rawBody).toBeTruthy()
    }, 15000)

    test(`trace emitted for CodingAgentChat on ${label}`, async () => {
      harness!.traces.clear()
      const resolved = await harness!.resolveTarget(target)
      await harness!.runBoundCodingAgentChat(resolved.bound)

      expect(harness!.traces.traces.length).toBeGreaterThanOrEqual(1)
      const trace = harness!.traces.traces.at(-1)!
      assertCommon(trace, target)

      if (target.expectedDriver === 'openai-responses') {
        expect(trace.request.input).toBeTruthy()
      } else {
        expect(trace.request.messages ?? trace.request.input).toBeTruthy()
      }

      expect(trace.response.rawBody).toBeTruthy()
    }, 30000)
  }
})

function assertCommon(trace: any, target: { slot: string; model: { id: string; providerId: string } }) {
  expect(typeof trace.timestamp).toBe('string')
  expect(trace.model).toBe(target.model.id)
  expect(trace.provider).toBe(target.model.providerId)
  expect(trace.slot).toBe(target.slot)
  expect(typeof trace.durationMs).toBe('number')
  expect(trace.durationMs).toBeGreaterThanOrEqual(0)
  expect(trace.usage).toBeTruthy()
}