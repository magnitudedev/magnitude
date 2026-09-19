import { Clock, Effect, Redacted, Schema } from "effect"
import { LabApiError } from "./client"
import { EntraApplicationId, EntraTenantId } from "./entra-auth"
import { command, ProcessExecutor } from "./process"

const Response = Schema.Struct({ accessToken: Schema.NonEmptyString, expires_on: Schema.Number, tenant: Schema.UUID })
export const entraClientToken = (tenant: typeof EntraTenantId.Type, application: typeof EntraApplicationId.Type, executable = "az") => Effect.gen(function* () {
  const executor = yield* ProcessExecutor
  const acquire = Effect.gen(function* () {
    const result = yield* command(executable, ["account", "get-access-token", "--tenant", tenant,
      "--scope", `api://${application}/Lab.Access`, "--output", "json", "--only-show-errors"], {
      timeoutMs: 30_000, maxOutputBytes: 64 * 1024,
    }).pipe(Effect.provideService(ProcessExecutor, executor),
      Effect.mapError(() => new LabApiError({ status: 0, message: "Azure login token request failed" })))
    if (result.exitCode !== 0) return yield* new LabApiError({ status: 0, message: "Azure login requires authentication or consent for the lab API" })
    const response = yield* Schema.decodeUnknown(Schema.parseJson(Response))(result.stdout).pipe(
      Effect.mapError(() => new LabApiError({ status: 0, message: "Invalid Azure login token response" })))
    if (response.tenant !== tenant || !Number.isFinite(response.expires_on) || response.expires_on * 1000 < (yield* Clock.currentTimeMillis) + 90_000) return yield* new LabApiError({ status: 0, message: "Azure token has an incorrect tenant or insufficient lifetime" })
    return Redacted.make(response.accessToken)
  })
  return yield* Effect.cachedWithTTL(acquire, "60 seconds")
})
