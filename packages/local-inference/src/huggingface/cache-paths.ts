import type * as Path from "@effect/platform/Path"
import type { HuggingFaceArtifactFile, HuggingFaceRemoteContentIdentity } from "./contracts"

export const huggingFaceRepositoryFolder = (repository: string): string => `models--${repository.replace("/", "--")}`

export const remoteContentKey = (content: HuggingFaceRemoteContentIdentity): string => {
  switch (content._tag) {
    case "LfsSha256": return content.sha256
    case "Xet": return content.hash
    case "Git": return content.oid
  }
}

export const snapshotPath = (path: Path.Path, cacheRoot: string, repository: string, commit: string, file: HuggingFaceArtifactFile): string =>
  path.join(cacheRoot, huggingFaceRepositoryFolder(repository), "snapshots", commit, file.path)

export const blobPath = (path: Path.Path, cacheRoot: string, repository: string, file: HuggingFaceArtifactFile): string =>
  path.join(cacheRoot, huggingFaceRepositoryFolder(repository), "blobs", remoteContentKey(file.content))

export const isWithin = (path: Path.Path, root: string, candidate: string): boolean => {
  const relative = path.relative(root, candidate)
  return relative === "" || (relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative))
}
