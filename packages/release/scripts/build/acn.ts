import {
  mkdir,
  readdir,
  readFile,
  writeFile,
} from "node:fs/promises"
import { relative, resolve } from "node:path"
import { downloadRg } from "../../../ripgrep/src/download"
import {
  bunTargetToRipgrepTarget,
  getTargetInfo,
} from "../../../../scripts/release-target"
import { run } from "./common"
import { ACN_EXECUTABLE_NAME } from "../../src/executables"
import { compileAppleBun, runAppleBuild } from "../apple/compile-bun"

const PROJECT_ROOT = resolve(import.meta.dir, "../../../..")
const RG_EMBED = resolve(PROJECT_ROOT, "packages/ripgrep/src/rg-embed.ts")
const WEB_APP_EMBED = resolve(PROJECT_ROOT, "packages/acn/src/web-app-embed.ts")
const WEB_ROOT = resolve(PROJECT_ROOT, "web")
const WEB_DIST = resolve(WEB_ROOT, "dist")

const listFiles = async (directory: string): Promise<ReadonlyArray<string>> =>
  (await Promise.all((await readdir(directory, { withFileTypes: true })).map(entry => {
    const path = resolve(directory, entry.name)
    return entry.isDirectory() ? listFiles(path) : Promise.resolve([path])
  }))).flat()

/** Builds the browser app and embeds every file of it in the service executable for the duration of `use`. */
const withWebAppEmbed = async <A>(use: () => Promise<A>): Promise<A> => {
  await run(["bun", "run", "build"], { cwd: WEB_ROOT })
  const files = [...await listFiles(WEB_DIST)].sort()
  const original = await readFile(WEB_APP_EMBED, "utf8")
  const embedDirectory = resolve(WEB_APP_EMBED, "..")
  const imports = files.map((file, index) => `// @ts-expect-error Bun resolves this file through its compile-time file loader.\nimport file${index} from ${JSON.stringify(relative(embedDirectory, file).split("\\").join("/"))} with { type: "file" }`)
  const entries = files.map((file, index) => `  { path: ${JSON.stringify(relative(WEB_DIST, file).split("\\").join("/"))}, file: file${index} as string },`)
  await writeFile(WEB_APP_EMBED, `${imports.join("\n")}\n\nexport const embeddedWebApp: ReadonlyArray<{ readonly path: string; readonly file: string }> = [\n${entries.join("\n")}\n]\n`)
  try {
    return await use()
  } finally {
    await writeFile(WEB_APP_EMBED, original)
  }
}

const withRipgrepEmbed = async <A>(
  windows: boolean,
  use: () => Promise<A>,
): Promise<A> => {
  const original = await readFile(RG_EMBED, "utf8")
  const binary = windows ? "rg.exe" : "rg"
  await writeFile(
    RG_EMBED,
    `// @ts-expect-error Bun resolves this executable through its compile-time file loader.\n` +
      `export { default as rgPath } from "../bin/${binary}" with { type: "file" };\n`,
  )
  try {
    return await use()
  } finally {
    await writeFile(RG_EMBED, original)
  }
}

export const buildAcnBinary = async (target: string): Promise<string> => {
  const info = getTargetInfo(target)
  const binary = resolve(
    PROJECT_ROOT,
    "bin",
    `${ACN_EXECUTABLE_NAME}${info.executableExt}`,
  )
  await mkdir(resolve(PROJECT_ROOT, "bin"), { recursive: true })
  await downloadRg(
    resolve(PROJECT_ROOT, "packages/ripgrep/bin"),
    bunTargetToRipgrepTarget(target),
  )
  if (info.platform === "darwin") return withWebAppEmbed(() => withRipgrepEmbed(false, () => runAppleBuild(compileAppleBun(resolve(PROJECT_ROOT, "packages/acn/src/binary.ts"), binary, target, "acn"))))
  await withWebAppEmbed(() => withRipgrepEmbed(info.platform === "windows", () =>
    run([
      process.execPath,
      "build",
      resolve(PROJECT_ROOT, "packages/acn/src/binary.ts"),
      "--compile",
      `--target=${target}`,
      `--outfile=${binary}`,
    ], { cwd: PROJECT_ROOT }),
  ))
  return binary
}
