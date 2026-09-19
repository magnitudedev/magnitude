import { Effect, Either } from "effect"
import { generateKeyPairSync } from "node:crypto"
import { expect, it } from "vitest"
import { decodeUpdateConfiguration } from "./configuration"

const publisher = generateKeyPairSync("ed25519")
const config = { origin: "https://magnitude.dev", acceptance: false, keyId: "publisher",
  publicKey: publisher.publicKey.export({ type: "spki", format: "pem" }).toString() }
const privateDelivery = { _tag: "PrivateAcceptance", origin: "https://private-lab.example" }
const decode = (input: unknown, build: boolean) => Effect.runPromise(Effect.either(decodeUpdateConfiguration(input, build)))

it("defaults both production and existing acceptance configurations to GitHub", async () => {
  for (const acceptance of [false, true]) {
    const result = await Effect.runPromise(decodeUpdateConfiguration({ ...config, acceptance,
      origin: acceptance ? "https://acceptance.example" : config.origin }, acceptance))
    expect(result.artifactDelivery).toEqual({ _tag: "Github" })
    expect(result.trustedPublishers.get(config.keyId)?.export({ type: "spki", format: "pem" })).toBe(config.publicKey)
  }
})
it("admits a private artifact origin only in a matching acceptance build", async () => {
  const input = { ...config, acceptance: true, origin: "https://acceptance.example", artifactDelivery: privateDelivery }
  const result = await Effect.runPromise(decodeUpdateConfiguration(input, true))
  expect(result.artifactDelivery).toEqual(privateDelivery)
  expect(Either.isLeft(await decode(input, false))).toBe(true)
  expect(Either.isLeft(await decode({ ...input, acceptance: false }, false))).toBe(true)
  expect(Either.isLeft(await decode(config, true))).toBe(true)
})
it.each(["https://magnitude.dev", "https://api.magnitude.dev", "http://localhost:1234", "https://lab.example/path", "https://lab.example?secret=1"])("rejects acceptance check origin %s", async origin => {
  expect(Either.isLeft(await decode({ ...config, origin, acceptance: true }, true))).toBe(true)
})
it("rejects malformed publisher keys, empty identities and unknown configuration", async () => {
  for (const patch of [{ publicKey: "not a public key" }, { keyId: "" }, { artifactOrigin: "https://typo.example" }]) {
    expect(Either.isLeft(await decode({ ...config, ...patch }, false))).toBe(true)
  }
})
