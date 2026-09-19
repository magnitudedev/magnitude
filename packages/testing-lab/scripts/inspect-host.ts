import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Console, Effect, Schema } from "effect"
import { findTarget } from "../src/catalog"
import { TargetId } from "../src/domain"
import { HostObservation } from "../src/hardware"
import { HostInspector, HostInspectorLive } from "../src/host-inspector"
import { ProcessExecutorLive } from "../src/process"

BunRuntime.runMain(Effect.gen(function* () {
  const target = yield* findTarget(TargetId.make(yield* Config.string("LAB_INSPECT_TARGET")))
  const inspector = yield* HostInspector
  const observation = yield* inspector.inspect(target)
  yield* Console.log(yield* Schema.encode(Schema.parseJson(HostObservation))(observation))
}).pipe(Effect.provide(HostInspectorLive), Effect.provide([BunContext.layer, ProcessExecutorLive])))
