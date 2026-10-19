import { Option } from "effect"
import { describe, expect, it } from "vitest"
import { HuggingFaceAccessRequired } from "@magnitudedev/sdk"
import { ADDED_MODEL_ID, makeAddedModel, makeInstalledCatalogModel, TEST_REPOSITORY } from "./fixtures/inference"
import { downloadFailureText } from "./hugging-face-access"
import { renderModelDetail, renderModelsStatus } from "./models-status-runtime"
import { renderCommandError } from "./output"

const SIGN_IN = "Sign in to Hugging Face to use this model. Run `magnitude hf login`."
const ACCEPT_TERMS = `Accept this model's terms on Hugging Face: https://huggingface.co/${TEST_REPOSITORY} — then try again. Some models also need the author's approval.`

describe("a gated repository refused by Hugging Face", () => {
  it.each([
    ["SignIn", SIGN_IN],
    ["AcceptTerms", ACCEPT_TERMS],
  ] as const)("tells the person what %s needs, in text and JSON", (access, message) => {
    const error = new HuggingFaceAccessRequired({ repository: TEST_REPOSITORY, access, message: "PRIVATE service wording" })
    expect(renderCommandError(error, false)).toBe(`${message}\n`)
    expect(JSON.parse(renderCommandError(error, true))).toEqual({ error: { message, code: "hugging_face_access_required" } })
  })

  it("words a failed gated download, naming the repository's page when it is known", () => {
    expect(downloadFailureText({ _tag: "HuggingFaceAccessRequired", access: "SignIn" }, Option.none())).toBe(SIGN_IN)
    expect(downloadFailureText({ _tag: "HuggingFaceAccessRequired", access: "AcceptTerms" }, Option.some(TEST_REPOSITORY))).toBe(ACCEPT_TERMS)
    expect(downloadFailureText({ _tag: "HuggingFaceAccessRequired", access: "AcceptTerms" }, Option.none()))
      .toBe("Accept this model's terms on Hugging Face, then try again. Some models also need the author's approval.")
    expect(downloadFailureText({ _tag: "SourceUnavailable" }, Option.some(TEST_REPOSITORY))).toBe("This model is not available from its source.")
  })

  it("shows a gated download failure in models status with the model's repository", () => {
    const model = makeAddedModel({
      acquisitionState: { _tag: "InstallFailed", failure: { _tag: "HuggingFaceAccessRequired", access: "AcceptTerms" } },
    })
    const list = renderModelsStatus({ models: [model], unsupported: [], hidden: [], all: false })
    expect(list).toContain(`Failed - ${ACCEPT_TERMS}`)
    expect(list).toContain(ADDED_MODEL_ID)
    const detail = renderModelDetail({ model, models: [model], drafters: { _tag: "NotApplicable" }, hidden: false })
    expect(detail).toMatch(new RegExp(`Reason\\s+${ACCEPT_TERMS.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`))
  })

  it("asks a catalog model's failed download to sign in", () => {
    const installed = makeInstalledCatalogModel()
    const model = { ...installed, acquisitionState: { _tag: "InstallFailed" as const, failure: { _tag: "HuggingFaceAccessRequired" as const, access: "SignIn" as const } } }
    expect(renderModelsStatus({ models: [model], unsupported: [], hidden: [], all: false })).toContain(`Failed - ${SIGN_IN}`)
  })
})
