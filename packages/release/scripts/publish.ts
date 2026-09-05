import { createHash } from "node:crypto"
import { readdir, readFile, stat } from "node:fs/promises"
import { resolve } from "node:path"
import { fromMarkdown } from "mdast-util-from-markdown"
import { Effect, Schema } from "effect"
import { BunContext } from "@effect/platform-bun"
import { prerelease as semverPrerelease } from "semver"
import { verifyPublishedPlugins } from "../src/plugin-artifacts"
import { ReleaseManifestSchema } from "../src/contracts"
import { fileSha256 } from "./build/common"
import {
  findGithubRelease,
  type GithubRelease,
  type GithubReleaseAsset,
} from "./github-release"

const required = (name: string): string => {
  const value = process.env[name]?.trim()
  if (!value) throw new Error(`${name} is required`)
  return value
}

const repository = required("GITHUB_REPOSITORY")
const token = required("GITHUB_TOKEN")
const headers = {
  accept: "application/vnd.github+json",
  authorization: `Bearer ${token}`,
  "x-github-api-version": "2022-11-28",
}

const github = async <A>(
  path: string,
  init: RequestInit = {},
  allowNotFound = false,
): Promise<A | undefined> => {
  const response = await fetch(`https://api.github.com${path}`, {
    ...init,
    headers: { ...headers, ...init.headers },
    signal: AbortSignal.timeout(60_000),
  })
  if (allowNotFound && response.status === 404) return undefined
  if (!response.ok) {
    throw new Error(`GitHub ${init.method ?? "GET"} ${path} returned HTTP ${response.status}`)
  }
  if (response.status === 204) return undefined
  return response.json() as Promise<A>
}

const assetSha256 = async (asset: GithubReleaseAsset): Promise<string> => {
  if (asset.digest?.startsWith("sha256:")) return asset.digest.slice("sha256:".length)
  const response = await fetch(asset.url, {
    headers: { ...headers, accept: "application/octet-stream" },
    redirect: "follow",
    signal: AbortSignal.timeout(10 * 60_000),
  })
  if (!response.ok) {
    throw new Error(`unable to authenticate existing GitHub asset ${asset.name}`)
  }
  const hash = createHash("sha256")
  for await (const chunk of response.body!) hash.update(chunk)
  return hash.digest("hex")
}

const candidate = resolve(process.argv[2] ?? "release-candidate")
const manifest = Schema.decodeUnknownSync(
  Schema.parseJson(ReleaseManifestSchema),
)(await readFile(resolve(candidate, "magnitude-release.json"), "utf8"))
const changelog = await readFile(
  resolve(import.meta.dir, "../../launcher/CHANGELOG.md"),
  "utf8",
)
const changelogTree = fromMarkdown(changelog)
const releaseHeadingIndex = changelogTree.children.findIndex((node) =>
  node.type === "heading" &&
  node.depth === 2 &&
  node.children.length === 1 &&
  node.children[0]?.type === "text" &&
  node.children[0].value === manifest.version
)
if (releaseHeadingIndex < 0) {
  throw new Error(`CHANGELOG.md has no entry for ${manifest.version}`)
}
const releaseHeading = changelogTree.children[releaseHeadingIndex]!
const nextReleaseHeading = changelogTree.children
  .slice(releaseHeadingIndex + 1)
  .find((node) => node.type === "heading" && node.depth === 2)
const notesStart = releaseHeading.position?.start.offset
const notesEnd = nextReleaseHeading?.position?.start.offset ?? changelog.length
if (notesStart === undefined) {
  throw new Error("CHANGELOG.md release entry has no source position")
}
const releaseNotes = changelog.slice(notesStart, notesEnd).trim()
const sourceCommit = required("MAGNITUDE_SOURCE_COMMIT")
if (manifest.sourceCommit !== sourceCommit) {
  throw new Error("candidate source commit differs from the workflow commit")
}
const prerelease = semverPrerelease(manifest.version) !== null
if (!prerelease) await Effect.runPromise(verifyPublishedPlugins(manifest.plugins, resolve(import.meta.dir, "../../..")).pipe(Effect.provide(BunContext.layer)))
const expectedNames = new Set([
  "magnitude-release.json",
  ...manifest.artifacts.map((artifact) => artifact.filename),
])
const localNames = new Set(await readdir(candidate))
if (
  localNames.size !== expectedNames.size ||
  [...localNames].some((name) => !expectedNames.has(name))
) {
  throw new Error("candidate directory differs from the release graph")
}
let release = await findGithubRelease(
  repository,
  token,
  manifest.tag,
  sourceCommit,
)
if (!release) {
  release = await github<GithubRelease>(`/repos/${repository}/releases`, {
    method: "POST",
    body: JSON.stringify({
      tag_name: manifest.tag,
      target_commitish: sourceCommit,
      name: manifest.tag,
      body: releaseNotes,
      draft: true,
      prerelease,
    }),
  })
}
if (
  !release ||
  !release.draft ||
  release.tag_name !== manifest.tag ||
  release.target_commitish !== sourceCommit
) {
  throw new Error("GitHub release is not the exact candidate draft")
}

const local = new Map(await Promise.all([...expectedNames].map(async (name) => {
  const path = resolve(candidate, name)
  const info = await stat(path)
  return [name, {
    path,
    bytes: Number(info.size),
    sha256: await fileSha256(path),
  }] as const
})))

// A draft is private and tied to this exact commit/version. Replacing its assets makes an
// interrupted upload genuinely resumable without pretending native builds are reproducible.
for (const asset of release.assets) {
  await github(`/repos/${repository}/releases/assets/${asset.id}`, {
    method: "DELETE",
  })
}
const uploadUrl = release.upload_url.slice(0, release.upload_url.indexOf("{"))
for (const [name, value] of local) {
  const response = await fetch(`${uploadUrl}?name=${encodeURIComponent(name)}`, {
    method: "POST",
    headers: {
      ...headers,
      "content-type": "application/octet-stream",
    },
    body: Bun.file(value.path),
    signal: AbortSignal.timeout(30 * 60_000),
  })
  if (!response.ok) {
    throw new Error(`upload of ${name} returned HTTP ${response.status}`)
  }
}

release = await github<GithubRelease>(
  `/repos/${repository}/releases/${release.id}`,
)
if (!release || release.assets.length !== local.size) {
  throw new Error("GitHub did not record the complete candidate")
}
for (const asset of release.assets) {
  const expected = local.get(asset.name)
  if (
    !expected ||
    asset.size !== expected.bytes ||
    await assetSha256(asset) !== expected.sha256
  ) {
    throw new Error(`uploaded asset ${asset.name} failed verification`)
  }
}

try {
  await github(`/repos/${repository}/releases/${release.id}`, {
    method: "PATCH",
    body: JSON.stringify({
      draft: false,
      prerelease,
      make_latest: prerelease ? "false" : "true",
    }),
  })
} catch {
  // A lost response is ambiguous. Resolve it from GitHub rather than assuming publication failed.
}
const published = await github<GithubRelease>(
  `/repos/${repository}/releases/tags/${encodeURIComponent(manifest.tag)}`,
)
if (
  !published ||
  published.draft ||
  published.tag_name !== manifest.tag ||
  published.target_commitish !== sourceCommit
) {
  throw new Error("GitHub release did not become public at the expected commit")
}
