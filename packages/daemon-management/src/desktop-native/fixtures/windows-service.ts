import assert from "node:assert/strict"
import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { isAbsolute, join } from "node:path"
import { Effect, Layer, Option, Stream } from "effect"
import { nativeWindowsJobOwnerLayer, nativeWindowsPrivatePipesLayer } from "@magnitudedev/utils/windows-native"
import { makeWindowsOwnedChildSpawner } from "../windows-owned-child"

// Run this fixture under Node and Bun against the compiled service, never a fake child.
const [service, addon, mode] = process.argv.slice(2)
const acknowledge = mode !== "--withhold-ack"
assert.equal(process.platform, "win32", "Requires native Windows")
assert.ok(service && isAbsolute(service), "Requires an absolute compiled service path")
assert.ok(addon && isAbsolute(addon), "Requires an absolute native addon path")

const run = Effect.acquireUseRelease(
  Effect.tryPromise(() => mkdtemp(join(tmpdir(), "Magnitude terminal health "))),
  root => Effect.scoped(Effect.gen(function* () {
    const spawner = yield* makeWindowsOwnedChildSpawner
    const child = yield* spawner.spawn({
      executable: service, arguments: ["serve", "--data-dir", root, "--port", "0"],
      environment: { ...process.env, MAGNITUDE_NATIVE_HOST: addon, MAGNITUDE_ICN_PATH: join(root, "absent-engine.json") },
    })
    let terminal = false
    yield* child.events.pipe(Stream.runForEach(event => Effect.gen(function* () {
      if (event._tag === "Booted") return yield* child.send({ _tag: "Start" })
      if (event.health.state._tag !== "Stopping") return
      const state = event.health.state
      assert.equal(state.reason, "startup-failed")
      assert.ok(Option.isSome(state.safeDetail))
      const detail = Option.getOrThrow(state.safeDetail)
      assert.match(detail, /inference server binary was not found/)
      assert.ok(!detail.includes("\n") && detail.length <= 500)
      terminal = true
      if (acknowledge) yield* child.send({ _tag: "StoppingObserved" })
    })), Effect.timeout("60 seconds"))
    assert.ok(terminal, "Final stopping health must arrive before control EOF")
    assert.equal(yield* child.exit.pipe(Effect.timeout("10 seconds")), 1)
    yield* child.stop
    yield* Effect.logInfo("PASS compiled service delivers terminal health before exit and retires its job")
  })),
  root => Effect.promise(() => rm(root, { recursive: true, force: true })),
).pipe(Effect.provide(Layer.merge(nativeWindowsJobOwnerLayer(addon), nativeWindowsPrivatePipesLayer(addon))))

Effect.runPromise(run).catch(error => { console.error(error); process.exitCode = 1 })
