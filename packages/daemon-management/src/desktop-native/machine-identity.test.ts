import { createRequire } from "node:module"
import { fileURLToPath } from "node:url"
import { Effect } from "effect"
import { expect, it } from "vitest"
import { readMachineIdentity } from "./machine-identity"

it.each([
  [{ manufacturer: "Apple", model: "Mac16,5" }, { _tag: "Identified", manufacturer: "Apple", model: "Mac16,5" }],
  [{ manufacturer: "Dell Inc.\n", model: " Precision 3680 \u0000" }, { _tag: "Identified", manufacturer: "Dell Inc.", model: "Precision 3680" }],
  [{ manufacturer: "ASUS", model: "System Product Name" }, { _tag: "Unavailable" }],
  [{ manufacturer: "To Be Filled By O.E.M.", model: "Board" }, { _tag: "Unavailable" }],
  [{ manufacturer: "Apple", model: "" }, { _tag: "Unavailable" }],
  [{ manufacturer: "Apple", model: 42 }, { _tag: "Unavailable" }],
  [{ manufacturer: "Apple", model: "x".repeat(256) }, { _tag: "Unavailable" }],
  [null, { _tag: "Unavailable" }],
])("validates firmware identity without inventing model names %#", async (raw, expected) => {
  expect(await Effect.runPromise(readMachineIdentity(() => ({ machineIdentity: () => raw })))).toEqual(expected)
})
it("keeps unavailable native identity out of the application error path", async () => {
  expect(await Effect.runPromise(readMachineIdentity(() => { throw new Error("Not supported") }))).toEqual({ _tag: "Unavailable" })
})
it.skipIf(process.platform !== "darwin")("reads the real Mac model through the packaged native boundary", async () => {
  const addon = fileURLToPath(new URL("../../dist/native/darwin-arm64/desktop-host.node", import.meta.url))
  const result = await Effect.runPromise(readMachineIdentity(() => createRequire(import.meta.url)(addon)))
  expect(result).toMatchObject({ _tag: "Identified", manufacturer: "Apple", model: expect.stringMatching(/^(Mac|iMac)/) })
})
