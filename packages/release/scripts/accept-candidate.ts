import {
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises"
import { tmpdir } from "node:os"
import { resolve } from "node:path"
import { Option, Schema } from "effect"
import { ReleaseManifestSchema } from "../src/contracts"
import { releaseUrl } from "../src/acquisition"
import { currentHost } from "../src/targets"
import { run } from "./build/common"
import { prepareNpmCandidate } from "./prepare-npm"

const candidate = resolve(process.argv[2] ?? "release-candidate")
const manifest = Schema.decodeUnknownSync(
  Schema.parseJson(ReleaseManifestSchema),
)(await readFile(resolve(candidate, "magnitude-release.json"), "utf8"))
const cliArtifact = manifest.artifacts.find((artifact) =>
  artifact.kind === "cli" && Option.getOrUndefined(artifact.host) === currentHost()
)
if (!cliArtifact) {
  throw new Error(`candidate has no CLI artifact for ${currentHost()}`)
}

const routes = new Map(
  [
    "magnitude-release.json",
    "magnitude-release.json.sig",
    ...manifest.artifacts.map((artifact) => artifact.filename),
  ].map((name) => [
    new URL(releaseUrl("http://release.invalid", manifest.version, name)).pathname,
    name,
  ]),
)
const server = Bun.serve({
  port: 0,
  async fetch(request) {
    const name = routes.get(new URL(request.url).pathname)
    if (!name) return new Response("missing", { status: 404 })
    try {
      return new Response(await readFile(resolve(candidate, name)))
    } catch {
      return new Response("missing", { status: 404 })
    }
  },
})
const baseUrl = `http://127.0.0.1:${server.port}`
const root = await mkdtemp(resolve(tmpdir(), "magnitude-candidate-"))
const environment = (home: string) => ({
  ...process.env,
  HOME: home,
  USERPROFILE: home,
  MAGNITUDE_RELEASE_BASE_URL: baseUrl,
})
const invoke = async (
  command: readonly string[],
  directory: string,
  home: string,
): Promise<void> => {
  const output = (await run(command, {
    cwd: directory,
    env: environment(home),
  })).trim()
  if (output !== manifest.version) {
    throw new Error(`${command[0]} returned ${output}; expected ${manifest.version}`)
  }
}
try {
  const npmRoot = resolve(root, "npm")
  const bunRoot = resolve(root, "bun")
  await mkdir(npmRoot)
  await mkdir(bunRoot)
  await writeFile(resolve(npmRoot, "package.json"), "{}\n")
  await writeFile(resolve(bunRoot, "package.json"), "{}\n")
  const tarball = process.argv[3]
    ? resolve(process.argv[3])
    : (await prepareNpmCandidate(resolve(root, "package"))).tarball
  await run(["npm", "install", "--ignore-scripts", tarball], { cwd: npmRoot })
  await invoke(
    ["node", resolve(npmRoot, "node_modules/@magnitudedev/cli/bin/magnitude.js"), "--version"],
    npmRoot,
    resolve(root, "home-node"),
  )
  await invoke(
    ["npx", "--no-install", "magnitude", "--version"],
    npmRoot,
    resolve(root, "home-npx"),
  )
  await run(["bun", "add", "--ignore-scripts", tarball], { cwd: bunRoot })
  await invoke(
    ["bun", resolve(bunRoot, "node_modules/@magnitudedev/cli/bin/magnitude.js"), "--version"],
    bunRoot,
    resolve(root, "home-bun"),
  )
  await invoke(
    ["bunx", "--bun", "magnitude", "--version"],
    bunRoot,
    resolve(root, "home-bunx"),
  )
} finally {
  server.stop(true)
  await rm(root, { recursive: true, force: true })
}
