import { describe, expect, it } from "vitest"
import { redactSecrets } from "./redact-secrets"

describe("redactSecrets", () => {
  it("hides Hugging Face tokens and authorization credentials", () => {
    expect(redactSecrets("token hf_oauth_AbC123_x leaked")).toBe("token hf_*** leaked")
    expect(redactSecrets("Authorization: Bearer abc.def-ghi")).toBe("Authorization: Bearer ***")
    expect(redactSecrets(`{"authorization":"Bearer secret"}`)).toBe(`{"authorization":"Bearer ***"}`)
    expect(redactSecrets("authorization=basic dXNlcjpwYXNz rest")).toBe("authorization=basic *** rest")
  })

  it("leaves ordinary text alone", () => {
    expect(redactSecrets("downloaded owner/model-GGUF in 3s")).toBe("downloaded owner/model-GGUF in 3s")
  })
})
