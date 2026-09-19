import { BunRuntime } from "@effect/platform-bun"
import { Config, Console, Effect, Redacted, Schema } from "effect"
import { Authenticator } from "../src/api"
import { EntraAuthConfig, entraAuthenticator } from "../src/entra-auth"
import { entraClientToken } from "../src/entra-token"
import { Principal } from "../src/domain"
import { ProcessExecutorLive } from "../src/process"

// Real Azure CLI acquisition followed by the production Microsoft JWKS verifier.
// Print only the admitted identity; never persist or log the token.
BunRuntime.runMain(Effect.gen(function* () {
  const config = yield* Config.string("LAB_ENTRA_PROBE_CONFIG").pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(EntraAuthConfig))))
  const acquire = yield* entraClientToken(config.tenantId, config.applicationId)
  const token = yield* acquire
  const identity = yield* Authenticator.pipe(Effect.flatMap(auth => auth.authenticate(`Bearer ${Redacted.value(token)}`)),
    Effect.provide(entraAuthenticator(config)))
  yield* Console.log(yield* Schema.encode(Schema.parseJson(Principal))(identity))
}).pipe(Effect.provide(ProcessExecutorLive)))
