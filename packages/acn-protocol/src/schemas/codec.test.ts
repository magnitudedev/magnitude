import { Schema } from "effect"
import { describe, expect, it } from "vitest"
import { versioned } from "./codec"

const Current = Schema.Struct({ title: Schema.String })
// Version 1 named the field `name`; it decodes to the current shape.
const First = Schema.transform(Schema.Struct({ name: Schema.String }), Schema.typeSchema(Current), {
  strict: true,
  decode: ({ name }) => ({ title: name }),
  encode: ({ title }) => ({ name: title }),
})
const Stored = versioned({ version: 2, schema: Current }, [{ version: 1, schema: First }])

describe("versioned", () => {
  it("encodes at the current version", () => {
    expect(Schema.encodeSync(Stored)({ title: "Hi" })).toStrictEqual({ version: 2, value: { title: "Hi" } })
  })

  it("decodes every version it lists to the current shape", () => {
    expect(Schema.decodeUnknownSync(Stored)({ version: 2, value: { title: "Hi" } })).toStrictEqual({ title: "Hi" })
    expect(Schema.decodeUnknownSync(Stored)({ version: 1, value: { name: "Hi" } })).toStrictEqual({ title: "Hi" })
  })

  it("rejects a version it does not list, and a value that does not decode at its version", () => {
    expect(Schema.decodeUnknownEither(Stored)({ version: 3, value: { title: "Hi" } })._tag).toBe("Left")
    expect(Schema.decodeUnknownEither(Stored)({ version: 1, value: { title: "Hi" } })._tag).toBe("Left")
  })

  it("decodes its value with the caller's parse options", () => {
    const extra = { version: 2, value: { title: "Hi", extra: true } }
    expect(Schema.decodeUnknownSync(Stored)(extra)).toStrictEqual({ title: "Hi" })
    expect(Schema.decodeUnknownEither(Stored, { onExcessProperty: "error" })(extra)._tag).toBe("Left")
    expect(Schema.decodeUnknownEither(Stored, { onExcessProperty: "error" })({ version: 1, value: { name: "Hi", extra: true } })._tag)
      .toBe("Left")
  })
})
