import { expect, test } from "vitest"
import { Effect, Schema } from "effect"
import { createLocalJWKSet, exportJWK, generateKeyPair, SignJWT } from "jose"
import { Authenticator } from "../src/api"
import { EntraAuthConfig, entraAuthenticator } from "../src/entra-auth"

const tenant = "4581d4bf-a664-4a42-a66a-c842beeec9e7"
const app = "11111111-2222-4333-8444-555555555555"
const user = "7b68fd28-7906-4529-aab6-c550148572f1"
const config = Schema.decodeUnknownSync(EntraAuthConfig)({ tenantId: tenant, applicationId: app, users: [user] })
test("admits only allowed tenant users with delegated permission for the lab API", async () => {
  const { privateKey, publicKey } = await generateKeyPair("RS256")
  const keys = createLocalJWKSet({ keys: [await exportJWK(publicKey)] })
  const auth = (token: string) => Effect.runPromise(Authenticator.pipe(Effect.flatMap(service => service.authenticate(`Bearer ${token}`)),
    Effect.provide(entraAuthenticator(config, keys)), Effect.either))
  const sign = (claims: Record<string, unknown> = {}, audience = app, issuer = `https://login.microsoftonline.com/${tenant}/v2.0`) => new SignJWT({
    tid: tenant, oid: user, ver: "2.0", scp: "Lab.Access", ...claims,
  }).setProtectedHeader({ alg: "RS256" }).setIssuer(issuer).setAudience(audience).setSubject("user")
    .setIssuedAt().setNotBefore("0s").setExpirationTime("1h").sign(privateKey)
  expect(await auth(await sign())).toMatchObject({ _tag: "Right", right: { owner: `entra:${tenant}:${user}`, trust: "developer" } })
  for (const token of [await sign({ tid: app }), await sign({ oid: app }), await sign({ scp: "Other" }),
    await sign({ scp: undefined, roles: ["Lab.Access"] }), await sign({ ver: "1.0" }),
    await sign({}, "https://management.azure.com/"), await sign({}, app, `https://sts.windows.net/${tenant}/`)]) {
    expect(await auth(token)).toMatchObject({ _tag: "Left", left: { _tag: "Unauthorized" } })
  }
})
