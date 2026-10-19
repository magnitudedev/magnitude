import { Option } from "effect"
import { describe, expect, it } from "vitest"
import { modelDownloadFailureMessage } from "../local-models/failure-messages"
import { huggingFaceAccessMessage, huggingFaceAccountSummary } from "./account"

describe("Hugging Face account wording", () => {
  it("summarizes each account state without a token", () => {
    expect(huggingFaceAccountSummary({ _tag: "SignedIn", username: "ada", avatarUrl: Option.none() })).toBe("Signed in as ada")
    expect(huggingFaceAccountSummary({ _tag: "SigningIn", method: "Browser" })).toBe("Waiting for the browser…")
    expect(huggingFaceAccountSummary({ _tag: "SignedOut", reason: Option.some("SessionExpired") })).toBe("Your Hugging Face sign-in expired. Sign in again.")
    expect(huggingFaceAccountSummary({ _tag: "SignedOut", reason: Option.none() })).toMatch(/^Not signed in/)
  })

  it("words gated access the same for inspect and download failures", () => {
    expect(huggingFaceAccessMessage("SignIn")).toBe("Sign in to Hugging Face to use this model.")
    expect(modelDownloadFailureMessage({ _tag: "HuggingFaceAccessRequired", access: "AcceptTerms" }))
      .toBe("Accept this model's terms on Hugging Face, then try again. Some models also need the author's approval.")
    expect(huggingFaceAccessMessage("AcceptTerms")).not.toMatch(/HF_TOKEN/)
  })
})
