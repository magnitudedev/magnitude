import { appendFile, readFile } from "node:fs/promises"
import { resolve } from "node:path"
import { releaseTag } from "../src/contracts"
import { run } from "./build/common"
import { findGithubRelease } from "./github-release"

const PROJECT_ROOT = resolve(import.meta.dir, "../../..")

const required = (name: string, fallback?: string): string => {
  const value = process.env[name]?.trim() || fallback
  if (!value) throw new Error(`${name} is required`)
  return value
}

const github = async (path: string): Promise<unknown | undefined> => {
  const response = await fetch(`https://api.github.com${path}`, {
    headers: {
      accept: "application/vnd.github+json",
      authorization: `Bearer ${required("GITHUB_TOKEN")}`,
      "x-github-api-version": "2022-11-28",
    },
    signal: AbortSignal.timeout(30_000),
  })
  if (response.status === 404) return undefined
  if (!response.ok) {
    throw new Error(`GitHub preflight returned HTTP ${response.status}`)
  }
  return response.json()
}

const npm = async (version: string): Promise<unknown | undefined> => {
  const response = await fetch(
    `https://registry.npmjs.org/%40magnitudedev%2Fcli/${encodeURIComponent(version)}`,
    { signal: AbortSignal.timeout(30_000) },
  )
  if (response.status === 404) return undefined
  if (!response.ok) {
    throw new Error(`npm preflight returned HTTP ${response.status}`)
  }
  return response.json()
}

const packageJson = JSON.parse(
  await readFile(resolve(PROJECT_ROOT, "packages/launcher/package.json"), "utf8"),
) as { readonly version?: string }
const version = required("MAGNITUDE_RELEASE_VERSION", packageJson.version)
const sourceCommit = required("MAGNITUDE_SOURCE_COMMIT", process.env.GITHUB_SHA)
if (packageJson.version !== version) {
  throw new Error("requested version differs from packages/launcher/package.json")
}
if (!/^[a-f0-9]{40}$/.test(sourceCommit)) {
  throw new Error("source commit must be a full lowercase SHA")
}
const releaseCommit = required("MAGNITUDE_RELEASE_COMMIT", sourceCommit)
if (!/^[a-f0-9]{40}$/.test(releaseCommit)) {
  throw new Error("release commit must be a full lowercase SHA")
}
const releasePackageJson = JSON.parse(await run([
  "git",
  "show",
  `${releaseCommit}:packages/launcher/package.json`,
], { cwd: PROJECT_ROOT })) as { readonly version?: string }
if (releasePackageJson.version !== version) {
  throw new Error("Changesets release commit differs from the requested version")
}
const previousPackageJson = JSON.parse(await run([
  "git",
  "show",
  `${releaseCommit}^:packages/launcher/package.json`,
], { cwd: PROJECT_ROOT })) as { readonly version?: string }
if (!previousPackageJson.version) {
  throw new Error("release commit has no previous Changesets-owned CLI version")
}
if (previousPackageJson.version === version) {
  // A Version PR may change only plugins. It must still have changed the release plan.
  const planAt = (ref: string) => run([
    "git",
    "show",
    `${ref}:packages/release/release-plan.json`,
  ], { cwd: PROJECT_ROOT }).catch(() => undefined)
  const [before, after] = await Promise.all([planAt(`${releaseCommit}^`), planAt(releaseCommit)])
  if (after === undefined || before === after) {
    throw new Error("release commit changed neither the Changesets-owned CLI version nor the release plan")
  }
}
await run([
  "git",
  "merge-base",
  "--is-ancestor",
  releaseCommit,
  sourceCommit,
], { cwd: PROJECT_ROOT })

const repository = required("GITHUB_REPOSITORY")
const tag = releaseTag(version)
const [tagRef, release, npmVersion] = await Promise.all([
  github(`/repos/${repository}/git/ref/tags/${encodeURIComponent(tag)}`),
  findGithubRelease(
    repository,
    required("GITHUB_TOKEN"),
    tag,
    sourceCommit,
  ),
  npm(version),
])
const exactRelease = release?.tag_name === tag &&
  release.target_commitish === sourceCommit
let action: "release" | "publish-npm" | "complete"
if (npmVersion) {
  if (!exactRelease || release?.draft || !tagRef) {
    throw new Error("npm version does not have the exact public GitHub release")
  }
  action = "complete"
} else if (release?.draft) {
  if (!exactRelease || tagRef) {
    throw new Error("existing GitHub release is not the exact resumable draft")
  }
  action = "release"
} else if (release) {
  if (!exactRelease || !tagRef) {
    throw new Error("existing public GitHub release is inconsistent")
  }
  action = "publish-npm"
} else if (tagRef) {
  throw new Error("GitHub has an orphan release tag")
} else {
  action = "release"
}

if (process.env.MAGNITUDE_REQUIRE_RELEASE === "true" && action !== "release") {
  throw new Error("release state changed before publication")
}
if (action !== "complete") {
  required("NODE_AUTH_TOKEN")
  await run(["npm", "whoami", "--registry", "https://registry.npmjs.org"])
}

const output = process.env.GITHUB_OUTPUT
if (output) {
  await appendFile(output, [
    `action=${action}`,
    `version=${version}`,
    `source_commit=${sourceCommit}`,
    "",
  ].join("\n"))
} else {
  console.log(JSON.stringify({ action, version, tag, sourceCommit }, null, 2))
}
