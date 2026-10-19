import {
  DirectoryAccessDenied, DirectoryNotFound, DrafterNotUsable, HuggingFaceAccessRequired, InvalidDirectoryPath, InvalidModelReference, LocalModelMutationFailed,
  ModelRepositoryNotFound, ModelSourceUnavailable, ModelVersionNotRunnable, PathNotDirectory,
} from "@magnitudedev/sdk"
import { drafterNotUsableMessage, modelVersionNotRunnableMessage } from "@magnitudedev/client-common"
import type { NoticeContent } from "./error-notice"

/** A candidate failure in the user's terms; `access` is set when Hugging Face needs a sign-in or accepted terms first. */
export interface CandidateNotice extends NoticeContent {
  readonly retryable: boolean
  readonly access?: HuggingFaceAccessRequired
}

const unreachable = { title: "Couldn’t reach the Magnitude service", description: "Check the service on Status, then try again.", retryable: true }

const sourceUnavailableNotice = (error: ModelSourceUnavailable): CandidateNotice => {
  switch (error.reason) {
    case "ServiceUnreachable": return unreachable
    case "SourceUnreachable": return { title: "Couldn’t reach Hugging Face", description: "Check your internet connection, then try again.", retryable: true }
    case "Unreadable": return { title: "Magnitude couldn’t read this model", description: error.message, retryable: false }
    case "Unavailable": return { title: "This model isn’t available", description: "It may be private or removed. Check it on Hugging Face.", retryable: false }
  }
}

const accessNotice = (error: HuggingFaceAccessRequired): CandidateNotice => error.access === "SignIn"
  ? { title: "Sign in to Hugging Face to use this model", description: "This model is gated. Sign in, then try again.", retryable: true, access: error }
  : { title: "Accept this model’s terms on Hugging Face", description: "Open the model on Hugging Face and accept its terms, then try again. Some models also need the author’s approval.", retryable: true, access: error }

/**
 * A Hugging Face or file reference that couldn't be read or added, in the user's terms. Service text
 * stays out; Magnitude's own product messages for a repository Hugging Face won't show and a file it
 * couldn't read are shown as written.
 */
export const candidateErrorNotice = (error: unknown): CandidateNotice => {
  if (error instanceof InvalidModelReference) return { title: "Magnitude can’t open this model", description: "Paste a Hugging Face link, owner/repo, or owner/repo:QUANT.", retryable: false }
  if (error instanceof ModelRepositoryNotFound) return { title: "Magnitude can’t find this repository", description: error.message, retryable: false }
  if (error instanceof ModelSourceUnavailable) return sourceUnavailableNotice(error)
  if (error instanceof HuggingFaceAccessRequired) return accessNotice(error)
  if (error instanceof ModelVersionNotRunnable) return { title: "This version doesn’t run here", description: modelVersionNotRunnableMessage(error), retryable: false }
  if (error instanceof DrafterNotUsable) return { title: "This drafter can’t be used", description: drafterNotUsableMessage(error), retryable: false }
  if (error instanceof LocalModelMutationFailed) return { title: "Couldn’t add this model", description: "Nothing was changed. Try again.", retryable: error.retryable }
  return unreachable
}

/** Why a folder couldn't be added or removed. */
export const modelLocationErrorNotice = (error: unknown): NoticeContent => {
  if (error instanceof DirectoryNotFound) return { title: "This folder doesn’t exist", description: "Choose a folder on the computer running Magnitude." }
  if (error instanceof DirectoryAccessDenied) return { title: "Magnitude can’t read this folder", description: "Check its permissions, then try again." }
  if (error instanceof PathNotDirectory) return { title: "This isn’t a folder", description: "Choose a folder rather than a file." }
  if (error instanceof InvalidDirectoryPath) return { title: "This path isn’t valid", description: "Choose a folder from the list." }
  if (error instanceof LocalModelMutationFailed) return { title: "Your folders weren’t changed", description: "Try again." }
  return unreachable
}
