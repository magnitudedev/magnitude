import { createHash } from "node:crypto"
import { Option } from "effect"
import type { HuggingFaceArtifact, HuggingFaceRemoteContentIdentity } from "./contracts"
import { HuggingFaceArtifactId } from "./identity"

const contentKey = (content: HuggingFaceRemoteContentIdentity): string => {
  switch (content._tag) {
    case "LfsSha256": return `lfs:${content.sha256}`
    case "Xet": return `xet:${content.hash}`
    case "Git": return `git:${content.oid}`
  }
}

type ArtifactIdentityInput = Pick<HuggingFaceArtifact, "repository" | "commit" | "files" | "relationships">

export const makeHuggingFaceArtifactId = (artifact: ArtifactIdentityInput): HuggingFaceArtifactId => {
  const canonical = [
    artifact.repository,
    artifact.commit,
    ...[...artifact.files].sort((left, right) => left.path.localeCompare(right.path)).map((file) => `${file.path}\0${file.role}\0${Option.getOrElse(file.shardIndex, () => -1)}\0${file.sizeBytes}\0${contentKey(file.content)}`),
    ...[...artifact.relationships].sort((left, right) => `${left.fromPath}\0${left.toPath}`.localeCompare(`${right.fromPath}\0${right.toPath}`)).map((relationship) => `${relationship.kind}\0${relationship.fromPath}\0${relationship.toPath}`),
  ].join("\0")
  return HuggingFaceArtifactId.make(`hf_${createHash("sha256").update(canonical).digest("hex")}`)
}
