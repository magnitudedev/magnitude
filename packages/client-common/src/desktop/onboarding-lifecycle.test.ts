import { Registry } from "@effect-atom/atom-react"
import { Effect, Option, Queue, Stream } from "effect"
import { Client } from "@magnitudedev/effect-query"
import { describe, expect, it } from "vitest"
import type { CatalogLocalModel, Change } from "@magnitudedev/sdk"
import { DesktopOnboarding } from "./onboarding"
import { makeSetupModel, providerModelId } from "./fixtures/model"
import { clientServicesLayer, type ClientServices } from "../state/client-services"
import type { AcnClientRequirements } from "../state/agent-client"
import { MagnitudeOperations } from "../state/application-operations"
import { fakeAcnImplementationsLayer } from "../state/fake-acn-implementations"

const makeHarness = async () => {
  const calls: string[] = []
  const changes = Effect.runSync(Queue.unbounded<Change>())
  let model = makeSetupModel(false)
  let completed = false
  const request = (name: string, payload: unknown) => Effect.sync(() => {
    calls.push(name)
    if (name === "GetModelCatalog") return {
      _tag: "Ready", providers: [], failures: [],
      models: [{ _tag: "Local", product: model, offering: { _tag: "None" } }],
      localModelPreparation: { discovery: { complete: true, modelsFound: 0 }, assessment: { complete: true, settledModels: 1, totalModels: 1 } },
    }
    if (name === "GetOnboardingState") return { completed }
    if (name === "CompleteOnboarding") { completed = true; return {} }
    if (name === "SyncLocalModel") {
      expect(payload).toEqual({ modelId: providerModelId })
      model = { ...model, acquisitionState: { _tag: "Installing", progress: {
        stage: "downloading", completedBytes: 0, totalBytes: 1, bytesPerSecond: Option.none(),
      } } }
      return { outcome: "Started" }
    }
    if (name === "LoadLocalModel") {
      expect(payload).toEqual({ modelId: providerModelId })
      model = { ...model, acquisitionState: readyAcquisition() }
      return {}
    }
    throw new Error(`Unexpected onboarding RPC: ${name}`)
  })
  const client = Client.make<typeof MagnitudeOperations, AcnClientRequirements, never, ClientServices, never>(
    MagnitudeOperations,
    fakeAcnImplementationsLayer(request, name => name === "StreamChanges" ? Stream.fromQueue(changes) : Stream.never),
    operations => clientServicesLayer(operations),
  )
  const registry = Registry.make()
  const reference = client.runtime.atom(DesktopOnboarding)
  const unmount = registry.mount(reference)
  const service = await Effect.runPromise(Registry.getResult(registry, reference))
  return {
    calls, service, registry,
    completed: () => completed,
    setAcquisition: async (acquisitionState: CatalogLocalModel["acquisitionState"]) => {
      model = { ...model, acquisitionState }
      await Effect.runPromise(Queue.offer(changes, { operation: "GetModelCatalog" }))
    },
    dispose: () => { unmount(); registry.dispose() },
  }
}
const readyAcquisition = (): CatalogLocalModel["acquisitionState"] => ({
  _tag: "Installed",
  installation: { _tag: "Resolved", primaryPath: "/models/setup.gguf", installedBytes: 1, ownership: "Magnitude" },
  residencyState: { _tag: "Ready", allocation: { contextWindowTokens: 32768, parallelSequences: 1, physicalContextTokens: 32768, memoryDomains: [] } },
})
const waitEffect = (predicate: () => boolean): Effect.Effect<void> => Effect.suspend(() => predicate()
  ? Effect.void : Effect.sleep("1 millis").pipe(Effect.zipRight(waitEffect(predicate))))

const waitUntil = (predicate: () => boolean) => Effect.runPromise(waitEffect(predicate).pipe(Effect.timeout("3 seconds")))

describe("desktop onboarding scope", () => {
  it("is passive and permits explicit Skip without installing or loading", async () => {
    const harness = await makeHarness()
    try {
      expect(harness.calls).toEqual([])
      await Effect.runPromise(harness.service.finish)
      expect(harness.completed()).toBe(true)
      expect(harness.calls).toContain("CompleteOnboarding")
      expect(harness.calls).not.toContain("SyncLocalModel")
    } finally { harness.dispose() }
  })

  it("serializes setup, waits for installation, loads the exact model, and requires explicit Finish", async () => {
    const harness = await makeHarness()
    try {
      await Effect.runPromise(harness.service.select(providerModelId))
      await waitUntil(() => harness.calls.includes("SyncLocalModel"))
      await Effect.runPromise(harness.service.select(providerModelId))
      expect(harness.calls.filter(call => call === "SyncLocalModel")).toHaveLength(1)
      expect(harness.calls).not.toContain("LoadLocalModel")
      await expect(Effect.runPromise(harness.service.finish)).rejects.toThrow("Wait for setup")
      await harness.setAcquisition(makeSetupModel(true).acquisitionState)
      await waitUntil(() => harness.registry.get(harness.service.state)._tag === "Ready")
      expect(harness.calls.filter(call => call === "LoadLocalModel")).toHaveLength(1)
      expect(harness.completed()).toBe(false)
      await Effect.runPromise(harness.service.finish)
      expect(harness.completed()).toBe(true)
    } finally { harness.dispose() }
  })

  it("does not load after a cancelled download", async () => {
    const harness = await makeHarness()
    try {
      await Effect.runPromise(harness.service.select(providerModelId))
      await waitUntil(() => harness.calls.includes("SyncLocalModel"))
      await harness.setAcquisition({ _tag: "NotInstalled" })
      await waitUntil(() => harness.registry.get(harness.service.state)._tag === "Failed")
      expect(harness.calls).not.toContain("LoadLocalModel")
      expect(harness.completed()).toBe(false)
    } finally { harness.dispose() }
  })
})
