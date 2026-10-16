import {
  DirectoryAccessDenied, DirectoryNotFound, InvalidDirectoryPath, InvalidModelReference, LocalModelMutationFailed,
  ModelSourceUnavailable, PathNotDirectory,
} from "@magnitudedev/sdk"
import type { NoticeContent } from "./error-notice"

const unreachable = { title: "Couldn’t reach the Magnitude service", description: "Check the service on Status, then try again.", retryable: true }

/** A Hugging Face or file reference that couldn't be read or added, in the user's terms; service text stays out. */
export const candidateErrorNotice = (error: unknown): NoticeContent & { readonly retryable: boolean } => {
  if (error instanceof InvalidModelReference) return { title: "Magnitude can’t open this model", description: "Paste a Hugging Face link, owner/repo, or owner/repo:QUANT.", retryable: false }
  if (error instanceof ModelSourceUnavailable) return error.retryable
    ? { title: "Couldn’t reach Hugging Face", description: "Check your internet connection, then try again.", retryable: true }
    : { title: "This model isn’t available", description: "It may be private, gated, or removed. Check it on Hugging Face.", retryable: false }
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
