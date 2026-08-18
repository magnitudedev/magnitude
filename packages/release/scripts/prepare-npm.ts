import { createHash } from "node:crypto"
import { appendFile, mkdir, readFile, rm } from "node:fs/promises"
import { basename, resolve } from "node:path"
import { run } from "./build/common"

const PACKAGE_ROOT = resolve(import.meta.dir, "../../launcher")
const EXPECTED_FILES = [
  "package/README.md",
  "package/bin/magnitude.js",
  "package/package.json",
]

export const validateNpmCandidate = async (tarball: string): Promise<void> => {
  const listing = (await run(["tar", "-tzf", tarball]))
    .split("\n")
    .filter(Boolean)
    .sort()
  if (JSON.stringify(listing) !== JSON.stringify(EXPECTED_FILES)) {
    throw new Error(`npm candidate has unexpected files: ${listing.join(", ")}`)
  }
}

export const prepareNpmCandidate = async (
  output: string,
): Promise<{ readonly tarball: string; readonly integrity: string }> => {
  await rm(output, { recursive: true, force: true })
  await mkdir(output, { recursive: true, mode: 0o700 })
  const packed = JSON.parse(await run([
    "npm",
    "pack",
    "--json",
    "--pack-destination",
    output,
  ], { cwd: PACKAGE_ROOT })) as readonly { readonly filename?: string }[]
  if (packed.length !== 1 || !packed[0]?.filename) {
    throw new Error("npm pack did not produce exactly one candidate")
  }
  const tarball = resolve(output, basename(packed[0].filename))
  await validateNpmCandidate(tarball)
  const integrity =
    `sha512-${createHash("sha512").update(await readFile(tarball)).digest("base64")}`
  return { tarball, integrity }
}

if (import.meta.main) {
  const candidate = await prepareNpmCandidate(
    resolve(process.argv[2] ?? "release/npm-candidate"),
  )
  if (process.env.GITHUB_OUTPUT) {
    await appendFile(process.env.GITHUB_OUTPUT, [
      `tarball=${candidate.tarball}`,
      `integrity=${candidate.integrity}`,
      "",
    ].join("\n"))
  } else {
    console.log(JSON.stringify(candidate, null, 2))
  }
}
