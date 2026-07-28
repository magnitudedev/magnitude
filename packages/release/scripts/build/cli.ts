import { mkdir } from "node:fs/promises"
import { resolve } from "node:path"
import { getTargetInfo } from "../../../../scripts/release-target"
import { run } from "./common"

const PROJECT_ROOT = resolve(import.meta.dir, "../../../..")

export const buildCliBinary = async (target: string): Promise<string> => {
  const info = getTargetInfo(target)
  const nativePlatform = info.platform === "windows" ? "win32" : info.platform
  const binary = resolve(
    PROJECT_ROOT,
    "bin",
    `magnitude-cli${info.executableExt}`,
  )
  const trustedKeys = process.env.MAGNITUDE_RELEASE_TRUSTED_KEYS_JSON?.trim()
  if (!trustedKeys) {
    throw new Error("MAGNITUDE_RELEASE_TRUSTED_KEYS_JSON is required")
  }
  JSON.parse(trustedKeys)
  await mkdir(resolve(PROJECT_ROOT, "bin"), { recursive: true })
  await run([
    "bun",
    "build",
    resolve(PROJECT_ROOT, "cli/src/index.tsx"),
    "--compile",
    `--target=${target}`,
    `--outfile=${binary}`,
    "--external",
    "electron",
    "--external",
    "chromium-bidi",
    "--define",
    `process.platform=${JSON.stringify(nativePlatform)}`,
    "--define",
    `process.arch=${JSON.stringify(info.arch)}`,
    "--define",
    `MAGNITUDE_RELEASE_TRUSTED_KEYS_JSON=${JSON.stringify(trustedKeys)}`,
  ], { cwd: PROJECT_ROOT })
  if (info.platform === "darwin") {
    await run(["codesign", "--force", "--deep", "--sign", "-", binary])
  }
  return binary
}
