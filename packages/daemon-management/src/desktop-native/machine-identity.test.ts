import { createRequire } from "node:module"
import { fileURLToPath } from "node:url"
import { Effect, Option } from "effect"
import { expect, it } from "vitest"
import { machineFormFactor, readMachineIdentity } from "./machine-identity"
const raw = { manufacturer: "Apple", model: "Mac16,5", family: "", version: "", chassisType: 0 }
const read = (value: unknown) => Effect.runPromise(readMachineIdentity(() => ({ machineIdentity: () => value })))
it("normalizes public identity and optional firmware labels", async () => {
  expect(await read({ ...raw, manufacturer: " Lenovo\n", model: "21HD\u0000", family: "ThinkPad T14 Gen 4", version: "ThinkPad T14 Gen 4", chassisType: 10 })).toEqual({ _tag: "Identified", manufacturer: "Lenovo", model: "21HD", family: Option.some("ThinkPad T14 Gen 4"), version: Option.some("ThinkPad T14 Gen 4"), formFactor: "Portable" })
  expect(await read(raw)).toEqual({ _tag: "Identified", manufacturer: "Apple", model: "Mac16,5", family: Option.none(), version: Option.none(), formFactor: "Unknown" })
})
it.each(["", "System Product Name", "Default string", "To Be Filled By O.E.M.", "x".repeat(256)])("preserves enclosure type when model is unusable: %s", async model => {
  expect(await read({ ...raw, model, chassisType: 10 })).toEqual({ _tag: "Unavailable", formFactor: "Portable" })
})
it("drops placeholder optional labels", async () => {
  expect(await read({ ...raw, family: "System Family", version: "Default string" })).toMatchObject({ _tag: "Identified", family: Option.none(), version: Option.none() })
})
it.each([null, {}, { ...raw, model: 42 }, { ...raw, chassisType: "10" }])("tolerates malformed output %#", async value => {
  expect(await read(value)).toEqual({ _tag: "Unavailable", formFactor: "Unknown" })
})
it.each([[8,"Portable"],[9,"Portable"],[10,"Portable"],[14,"Portable"],[30,"Portable"],[31,"Portable"],[32,"Portable"],[3,"Desktop"],[7,"Desktop"],[13,"AllInOne"],[35,"MiniPc"],[36,"MiniPc"],[23,"Server"],[0,"Unknown"],[255,"Unknown"]] as const)("classifies enclosure %s as %s", (value, expected) => expect(machineFormFactor(value)).toBe(expected))
it("tolerates unavailable native bindings", async () => {
  expect(await Effect.runPromise(readMachineIdentity(() => { throw new Error("Not supported") }))).toEqual({ _tag: "Unavailable", formFactor: "Unknown" })
})
it.skipIf(process.platform !== "darwin")("reads the actual Mac through the packaged boundary", async () => {
  const addon = fileURLToPath(new URL("../../dist/native/darwin-arm64/desktop-host.node", import.meta.url))
  expect(await Effect.runPromise(readMachineIdentity(() => createRequire(import.meta.url)(addon)))).toMatchObject({ _tag: "Identified", manufacturer: "Apple", model: expect.stringMatching(/^(Mac|iMac)/), formFactor: "Unknown" })
})
