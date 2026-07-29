import { appendFile, readFile } from "node:fs/promises"
import { resolve } from "node:path"
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
  await readFile(resolve(PROJECT_ROOT, "packages/cli/package.json"), "utf8"),
) as { readonly version?: string }
const version = required("MAGNITUDE_RELEASE_VERSION", packageJson.version)
const sourceCommit = required("MAGNITUDE_SOURCE_COMMIT", process.env.GITHUB_SHA)
if (packageJson.version !== version) {
  throw new Error("requested version differs from packages/cli/package.json")
}
if (!/^[a-f0-9]{40}$/.test(sourceCommit)) {
  throw new Error("source commit must be a full lowercase SHA")
}
const previousPackageJson = JSON.parse(await run([
  "git",
  "show",
  `${sourceCommit}^:packages/cli/package.json`,
], { cwd: PROJECT_ROOT })) as { readonly version?: string }
if (!previousPackageJson.version || previousPackageJson.version === version) {
  throw new Error("release commit did not change the Changesets-owned CLI version")
}
required("NODE_AUTH_TOKEN")
await run(["npm", "whoami", "--registry", "https://registry.npmjs.org"])

const repository = required("GITHUB_REPOSITORY")
const tag = `@magnitudedev/cli@${version}`
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
if (npmVersion) throw new Error(`${version} already exists on npm`)
if (release) {
  if (
    release.draft !== true ||
    release.tag_name !== tag ||
    release.target_commitish !== sourceCommit ||
    tagRef
  ) {
    throw new Error("existing GitHub release is not the exact resumable draft")
  }
} else if (tagRef) {
  throw new Error("GitHub has an orphan release tag")
}

const output = process.env.GITHUB_OUTPUT
if (output) {
  await appendFile(output, [
    `version=${version}`,
    `source_commit=${sourceCommit}`,
    "",
  ].join("\n"))
} else {
  console.log(JSON.stringify({ version, tag, sourceCommit }, null, 2))
}
