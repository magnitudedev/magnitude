import { describe, expect, it } from "vitest"
import { Effect, Schema } from "effect"
import { createLocalJWKSet, exportJWK, generateKeyPair, SignJWT } from "jose"
import { Authenticator } from "../src/api"
import { GitHubAuthConfig, githubAuthenticator } from "../src/github-auth"

const config = Schema.decodeUnknownSync(GitHubAuthConfig)({ audience: "magnitude-lab", repositories: [{ repositoryId: "123", ownerId: "456" }] })
const issuer = "https://token.actions.githubusercontent.com"
describe("GitHub identity admission", () => {
  it("verifies signatures and binds untrusted identities to repository, run and attempt", async () => {
    const { privateKey, publicKey } = await generateKeyPair("RS256")
    const keys = createLocalJWKSet({ keys: [await exportJWK(publicKey)] })
    const authenticate = (token: string) => Effect.runPromise(Authenticator.pipe(Effect.flatMap(auth => auth.authenticate(`Bearer ${token}`)), Effect.provide(githubAuthenticator(config, keys)), Effect.either))
    const sign = (claims: Record<string, unknown> = {}, audience = "magnitude-lab", tokenIssuer = issuer, expiry = "5m") => new SignJWT({
      repository_id: "123", repository_owner_id: "456", run_id: "789", run_attempt: "1", event_name: "pull_request",
      trust: "developer", owner: "admin", ...claims,
    }).setProtectedHeader({ alg: "RS256" }).setIssuer(tokenIssuer).setAudience(audience).setSubject("repo:any:pull_request")
      .setIssuedAt().setNotBefore("0s").setExpirationTime(expiry).sign(privateKey)
    expect(await authenticate(await sign())).toMatchObject({ _tag: "Right", right: { owner: "github:123:789:1", trust: "untrusted-ci" } })
    expect(await authenticate(await sign({ run_attempt: "2" }))).toMatchObject({ _tag: "Right", right: { owner: "github:123:789:2" } })
    for (const token of [await sign({ repository_id: "999" }), await sign({ repository_owner_id: "999" }),
      await sign({ event_name: "pull_request_target" }), await sign({ run_id: "../admin" }), await sign({}, "other"),
      await sign({}, "magnitude-lab", "https://attacker.invalid"), await sign({}, "magnitude-lab", issuer, "-1m"), "not-a-jwt"]) {
      expect(await authenticate(token)).toMatchObject({ _tag: "Left", left: { _tag: "Unauthorized" } })
    }
    const other = await generateKeyPair("RS256")
    const forged = await new SignJWT({}).setProtectedHeader({ alg: "RS256" }).setIssuer(issuer).setAudience(config.audience).sign(other.privateKey)
    expect(await authenticate(forged)).toMatchObject({ _tag: "Left", left: { _tag: "Unauthorized" } })
    const noExpiry = await new SignJWT({}).setProtectedHeader({ alg: "RS256" }).setIssuer(issuer).setAudience(config.audience).setIssuedAt().sign(privateKey)
    expect(await authenticate(noExpiry)).toMatchObject({ _tag: "Left", left: { _tag: "Unauthorized" } })
  })
})
