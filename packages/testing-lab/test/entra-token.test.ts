import { Effect, Layer, Redacted, TestClock, TestContext } from "effect"
import { expect, test } from "vitest"
import { entraClientToken } from "../src/entra-token"
import { EntraApplicationId, EntraTenantId } from "../src/entra-auth"
import { ProcessExecutor } from "../src/process"
const tenant = EntraTenantId.make("4581d4bf-a664-4a42-a66a-c842beeec9e7")
const app = EntraApplicationId.make("11111111-2222-4333-8444-555555555555")
test("renews developer credentials using the explicit tenant and lab API scope", async () => {
  let issued = 0
  const executor = Layer.succeed(ProcessExecutor, { run: spec => Effect.sync(() => {
    expect(spec.args).toEqual(["account", "get-access-token", "--tenant", tenant, "--scope", `api://${app}/Lab.Access`, "--output", "json", "--only-show-errors"])
    return { exitCode: 0, stderr: "", stdout: JSON.stringify({ accessToken: `token-${++issued}`, expires_on: 3600, tenant }) }
  }) })
  await Effect.runPromise(Effect.gen(function* () {
    const acquire = yield* entraClientToken(tenant, app)
    expect(Redacted.value(yield* acquire)).toBe("token-1")
    expect(Redacted.value(yield* acquire)).toBe("token-1")
    yield* TestClock.adjust("61 seconds")
    expect(Redacted.value(yield* acquire)).toBe("token-2")
  }).pipe(Effect.provide(Layer.merge(executor, TestContext.TestContext))))
})
test("sanitizes CLI failures and refuses tokens for another tenant", async () => {
  for (const response of [{ exitCode: 1, stderr: "secret", stdout: "secret" },
    { exitCode: 0, stderr: "", stdout: JSON.stringify({ accessToken: "secret", expires_on: 3600, tenant: app }) }]) {
    const result = await Effect.runPromise(entraClientToken(tenant, app).pipe(Effect.flatten,
      Effect.provide(Layer.merge(Layer.succeed(ProcessExecutor, { run: () => Effect.succeed(response) }), TestContext.TestContext)), Effect.either))
    expect(result._tag).toBe("Left")
    expect(JSON.stringify(result)).not.toContain("secret")
  }
})
