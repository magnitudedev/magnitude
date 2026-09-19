import { Effect, Layer, Schema } from "effect"
import { createRemoteJWKSet, jwtVerify, type JWTVerifyGetKey } from "jose"
import { Authenticator, Unauthorized } from "./api"
import { OwnerId, type Principal } from "./domain"

const numericId = Schema.String.pipe(Schema.pattern(/^[1-9][0-9]{0,19}$/))
export const GitHubAuthConfig = Schema.Struct({
  audience: Schema.NonEmptyString.pipe(Schema.brand("LabOidcAudience")),
  repositories: Schema.NonEmptyArray(Schema.Struct({
    repositoryId: numericId.pipe(Schema.brand("GitHubRepositoryId")),
    ownerId: numericId.pipe(Schema.brand("GitHubOwnerId")),
  })),
})
export type GitHubAuthConfig = typeof GitHubAuthConfig.Type
const Claims = Schema.Struct({ repository_id: numericId, repository_owner_id: numericId,
  run_id: numericId, run_attempt: numericId,
  event_name: Schema.Literal("pull_request", "push", "workflow_dispatch", "merge_group"),
})
const issuer = "https://token.actions.githubusercontent.com"
/** The key resolver is injectable for cryptographic tests; production never uses a token-supplied URL. */
export const githubAuthenticator = (config: GitHubAuthConfig, keys: JWTVerifyGetKey = createRemoteJWKSet(
  new URL(`${issuer}/.well-known/jwks`), { timeoutDuration: 5_000, cooldownDuration: 30_000, cacheMaxAge: 600_000 },
)) => Layer.succeed(Authenticator, {
  authenticate: (header: string | undefined): Effect.Effect<Principal, Unauthorized> => Effect.gen(function* () {
    if (!header?.startsWith("Bearer ") || header.length > 32_768) return yield* new Unauthorized({})
    // JOSE is the cryptographic Promise boundary. Never log tokens or verification error payloads.
    const verified = yield* Effect.tryPromise({ try: () => jwtVerify(header.slice(7), keys, {
      issuer, audience: config.audience, algorithms: ["RS256"], requiredClaims: ["exp", "iat", "nbf", "sub"],
      maxTokenAge: "10 minutes", clockTolerance: 5,
    }), catch: () => new Unauthorized({}) })
    const claims = yield* Schema.decodeUnknown(Claims)(verified.payload).pipe(Effect.mapError(() => new Unauthorized({})))
    if (!config.repositories.some(repo => repo.repositoryId === claims.repository_id && repo.ownerId === claims.repository_owner_id)) return yield* new Unauthorized({})
    return { owner: OwnerId.make(`github:${claims.repository_id}:${claims.run_id}:${claims.run_attempt}`), trust: "untrusted-ci" }
  }),
})
