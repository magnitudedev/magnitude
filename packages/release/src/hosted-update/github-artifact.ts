import type { UpdateManifest } from "./manifest"

/** Repository ownership is trusted composition, never supplied by a download request. */
export const githubArtifactUrl = (manifest: UpdateManifest): string =>
  `https://github.com/magnitudedev/magnitude/releases/download/${manifest.tag.split("/").map(encodeURIComponent).join("/")}/${encodeURIComponent(manifest.artifact.filename)}`
