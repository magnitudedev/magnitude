import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Effect, Layer } from "effect"
import { AcnServerLayer } from "./server"

const program = Layer.launch(AcnServerLayer({ register: false, debug: false })).pipe(
  Effect.provide(BunContext.layer)
)
BunRuntime.runMain(program)
