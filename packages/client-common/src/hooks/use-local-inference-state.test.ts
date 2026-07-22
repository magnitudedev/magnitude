import { Atom, Registry, Result } from "@effect-atom/atom-react"
import { Schema } from "effect"
import { describe, expect, it } from "vitest"
import {
  IcnHardwareMirror,
  IcnInventoryMirror,
  ModelRecipesMirror,
} from "@magnitudedev/sdk"
import { makeLocalInferenceQueryAtom } from "./use-local-inference-state"

const hardware = Schema.decodeUnknownSync(IcnHardwareMirror.snapshotSchema)({
  revision: 1,
  state: {
    architecture: "arm64",
    captured_at: 1,
    cpu_model: "Apple M4 Max",
    enabled_backends: ["Metal"],
    logical_cores: 16,
    memory_domains: [],
    native_build: "test",
    platform: "macos",
    resident_memory: null,
    system_memory: {
      current_available_bytes: 32,
      total_bytes: 64,
    },
    topology_fingerprint: "test",
  },
})

const inventory = Schema.decodeUnknownSync(IcnInventoryMirror.snapshotSchema)({
  revision: 1,
  state: { object: "list", data: [] },
})

const recipes = Schema.decodeUnknownSync(ModelRecipesMirror.snapshotSchema)({
  revision: 1,
  state: { _tag: "Loading" },
})

describe("local inference query atom", () => {
  it("is initial until every mirror is available, then derives one coherent view", () => {
    const hardwareAtom = Atom.make<Result.Result<typeof hardware, never>>(Result.initial(true))
    const inventoryAtom = Atom.make<Result.Result<typeof inventory, never>>(Result.initial(true))
    const recipesAtom = Atom.make<Result.Result<typeof recipes, never>>(Result.initial(true))
    const combined = makeLocalInferenceQueryAtom(hardwareAtom, inventoryAtom, recipesAtom)
    const registry = Registry.make()

    expect(Result.isInitial(registry.get(combined))).toBe(true)
    registry.set(hardwareAtom, Result.success(hardware))
    registry.set(inventoryAtom, Result.success(inventory))
    expect(Result.isInitial(registry.get(combined))).toBe(true)
    registry.set(recipesAtom, Result.success(recipes))

    const result = registry.get(combined)
    expect(Result.isSuccess(result)).toBe(true)
    if (Result.isSuccess(result)) {
      expect(result.value).toMatchObject({
        host: { cpuModel: "Apple M4 Max" },
        choices: [],
        operations: [],
        recommendationState: { _tag: "Loading" },
      })
    }
    registry.dispose()
  })
})
