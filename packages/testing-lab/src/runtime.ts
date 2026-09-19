import { Effect } from "effect"
import { packageManager } from "../../../package.json"
import { InfrastructureFailure } from "./domain"

/** Match the application build's declared runtime, including when the driver is bundled remotely. */
export const assertRuntime = Effect.suspend(() => {
  const expected = packageManager.replace(/^bun@/, "")
  return process.versions.bun === expected ? Effect.void : Effect.fail(new InfrastructureFailure({
    operation: "runtime", message: `Testing requires Bun ${expected}; observed ${process.versions.bun ?? "a non-Bun runtime"}`,
  }))
})
