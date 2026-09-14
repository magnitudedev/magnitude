import type { UpdateManifest } from "./manifest"

/** Repository ownership is trusted composition, never supplied by a download request. */
export const githubArtifactUrl = (manifest: UpdateManifest): string =>
  `https://github.com/magnitudedev/magnitude/releases/download/${manifest.tag.split("/").map(encodeURIComponent).join("/")}/${encodeURIComponent(manifest.artifact.filename)}`

/** The release version resolves server-side; bytes remain authenticated by publisher proof. */
export const isGithubReleaseAssetUrl = (input: string): boolean => {
  try {
    const url = new URL(input)
    return url.protocol === "https:" && url.host === "github.com" && !url.username && !url.password
      && !url.search && !url.hash && url.href === input
      && /^\/magnitudedev\/magnitude\/releases\/download\/.+\/[^/]+$/.test(url.pathname)
      && !/%(?:2f|5c|00)/i.test(url.pathname)
  } catch { return false }
}

export const isGithubDeliveryUrl = (input: string): boolean => {
  if (isGithubReleaseAssetUrl(input)) return true
  try {
    const url = new URL(input)
    return url.protocol === "https:" && url.host === "release-assets.githubusercontent.com"
      && !url.username && !url.password && !url.hash && url.pathname.startsWith("/github-production-release-asset/")
  } catch { return false }
}
