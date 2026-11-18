import { describe, expect, it } from "vitest"
import { engineFailureSentence } from "./engine-failure"

describe("engineFailureSentence", () => {
  it("gives a code that names an inherited object property the unknown-code sentence", () => {
    for (const code of ["constructor", "toString", "__proto__", "hasOwnProperty"]) {
      expect(engineFailureSentence(code)).toBe(`The inference engine reported an error (${code}). Try again, or choose another model.`)
    }
  })
})
