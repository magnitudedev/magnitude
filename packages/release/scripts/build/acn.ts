import {
  mkdir,
  readFile,
  writeFile,
} from "node:fs/promises"
import { resolve } from "node:path"
import { downloadRg } from "../../../ripgrep/src/download"
import {
  bunTargetToRipgrepTarget,
  getTargetInfo,
} from "../../../../scripts/release-target"
import { run } from "./common"
import { ACN_EXECUTABLE_NAME } from "../../src/executables"

const PROJECT_ROOT = resolve(import.meta.dir, "../../../..")
const RG_EMBED = resolve(PROJECT_ROOT, "packages/ripgrep/src/rg-embed.ts")

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
  await withRipgrepEmbed(info.platform === "windows", () =>
    run([
      "bun",
      "build",
      resolve(PROJECT_ROOT, "packages/acn/src/binary.ts"),
      "--compile",
      `--target=${target}`,
      `--outfile=${binary}`,
    ], { cwd: PROJECT_ROOT }),
  )
  if (info.platform === "darwin") {
    await run(["codesign", "--force", "--deep", "--sign", "-", binary])
  }
  return binary
}
