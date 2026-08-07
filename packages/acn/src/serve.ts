import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Effect } from "effect"
import { launchAcnServer } from "./server"

const program = launchAcnServer({ debug: false }).pipe(
  Effect.provide(BunContext.layer),
)
BunRuntime.runMain(program)
