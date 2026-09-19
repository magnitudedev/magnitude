import { Effect, Layer, Schema } from "effect"
import { createRemoteJWKSet, jwtVerify, type JWTVerifyGetKey } from "jose"
import { Authenticator, Unauthorized } from "./api"
import { OwnerId, type Principal } from "./domain"

export const EntraTenantId = Schema.UUID.pipe(Schema.brand("LabEntraTenantId"))
export const EntraApplicationId = Schema.UUID.pipe(Schema.brand("LabEntraApplicationId"))
const ObjectId = Schema.UUID.pipe(Schema.brand("LabEntraObjectId"))
export const EntraAuthConfig = Schema.Struct({ tenantId: EntraTenantId, applicationId: EntraApplicationId,
  users: Schema.NonEmptyArray(ObjectId),
})
export type EntraAuthConfig = typeof EntraAuthConfig.Type
const Claims = Schema.Struct({ tid: EntraTenantId, oid: ObjectId, ver: Schema.Literal("2.0"), scp: Schema.String })
/** Only delegated access tokens for this API admit developers, never ARM or ID tokens. */
export const entraAuthenticator = (config: EntraAuthConfig, keys: JWTVerifyGetKey = createRemoteJWKSet(
  new URL(`https://login.microsoftonline.com/${config.tenantId}/discovery/v2.0/keys`),
  { timeoutDuration: 5_000, cooldownDuration: 30_000, cacheMaxAge: 600_000 },
)) => Layer.succeed(Authenticator, {
  authenticate: (header: string | undefined): Effect.Effect<Principal, Unauthorized> => Effect.gen(function* () {
    if (!header?.startsWith("Bearer ") || header.length > 32_768) return yield* new Unauthorized({})
    const verified = yield* Effect.tryPromise({ try: () => jwtVerify(header.slice(7), keys, {
      issuer: `https://login.microsoftonline.com/${config.tenantId}/v2.0`, audience: config.applicationId,
      algorithms: ["RS256"], requiredClaims: ["exp", "iat", "nbf", "sub"], clockTolerance: 5,
    }), catch: () => new Unauthorized({}) })
    const claims = yield* Schema.decodeUnknown(Claims)(verified.payload).pipe(Effect.mapError(() => new Unauthorized({})))
    if (claims.tid !== config.tenantId || !config.users.includes(claims.oid) || !claims.scp.split(" ").includes("Lab.Access")) return yield* new Unauthorized({})
    return { owner: OwnerId.make(`entra:${claims.tid}:${claims.oid}`), trust: "developer" }
  }),
})
