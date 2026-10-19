import { huggingFaceAccessMessage, modelDownloadFailureMessage } from "@magnitudedev/client-common"
import {
  huggingFaceRepositoryUrl,
  parseModelId,
  type DrafterReference,
  type HuggingFaceAccess,
  type ModelAcquisitionFailure,
  type ModelId,
} from "@magnitudedev/sdk"
import { Option } from "effect"

export const HUGGING_FACE_LOGIN_COMMAND = "magnitude hf login"

/** What a gated repository needs, with the step that provides it from a terminal. */
export const huggingFaceAccessText = (access: HuggingFaceAccess, repository: Option.Option<string>): string => {
  if (access === "SignIn") return `${huggingFaceAccessMessage(access)} Run \`${HUGGING_FACE_LOGIN_COMMAND}\`.`
  return Option.match(repository, {
    onNone: () => huggingFaceAccessMessage(access),
    onSome: (name) => `Accept this model's terms on Hugging Face: ${huggingFaceRepositoryUrl(name)} — then try again. Some models also need the author's approval.`,
  })
}

/** A failed download in the CLI's words; a gated repository names what to do about it. */
export const downloadFailureText = (failure: ModelAcquisitionFailure, repository: Option.Option<string>): string =>
  failure._tag === "HuggingFaceAccessRequired"
    ? huggingFaceAccessText(failure.access, repository)
    : modelDownloadFailureMessage(failure)

export const modelRepository = (modelId: ModelId): Option.Option<string> => {
  const parsed = parseModelId(modelId)
  return parsed._tag === "HuggingFace" ? Option.some(parsed.repositoryId) : Option.none()
}

export const drafterRepository = (reference: DrafterReference): Option.Option<string> =>
  reference._tag === "HuggingFace" ? Option.some(reference.repository) : Option.none()
