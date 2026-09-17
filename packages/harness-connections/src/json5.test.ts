import { expect, it } from "vitest"
import { json5Object, updateJson5 } from "./json5"

it("preserves JSON5 comments, unrelated values, and native quoting during field edits", () => {
  const source = "// user comment\n{ channel: { name: 'Keep me', }, models: { providers: {} }, }\n"
  const connected = updateJson5(source, [[["models", "providers", "magnitude"], { baseUrl: "http://localhost" }]])
  expect(json5Object(connected)).toMatchObject({ channel: { name: "Keep me" }, models: { providers: { magnitude: { baseUrl: "http://localhost" } } } })
  expect(connected).toContain("// user comment")
  expect(connected).toContain("name: 'Keep me'")
  const disconnected = updateJson5(connected, [[["models", "providers", "magnitude"], undefined]])
  expect(json5Object(disconnected)).toEqual(json5Object(source))
  expect(disconnected).toContain("// user comment")
  expect(updateJson5(source, [[["absent", "nested", "field"], undefined]])).toBe(source)
})

it("rejects malformed input and non-object roots before editing", () => {
  for (const source of ["{ broken", "[]", "null", "{__proto__: {polluted: true}}"])
    expect(() => updateJson5(source, [[["models"], {}]])).toThrow()
})
