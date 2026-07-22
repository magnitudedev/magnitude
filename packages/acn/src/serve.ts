import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Effect } from "effect"
import { launchAcnServer } from "./server"

const program = launchAcnServer({ register: false, debug: false }).pipe(
  Effect.provide(BunContext.layer),
)
BunRuntime.runMain(program)
