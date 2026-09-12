import { access, mkdir } from "node:fs/promises"
import { resolve } from "node:path"
import { getTargetInfo } from "../../../../scripts/release-target"
import { run } from "./common"
import { compileAppleBun, runAppleBuild } from "../apple/compile-bun"
import { signAppleCode } from "../apple/signing"

const PROJECT_ROOT = resolve(import.meta.dir, "../../../..")

export const buildCliBinary = async (target: string): Promise<string> => {
  const info = getTargetInfo(target)
  const nativePlatform = info.platform === "windows" ? "win32" : info.platform
  if (nativePlatform === process.platform && info.arch === process.arch) {
    await run([process.execPath, resolve(PROJECT_ROOT, "packages/daemon-management/scripts/build-native.ts")], { cwd: PROJECT_ROOT })
  }
  // Cross-compilation requires an already built addon for the selected target.
  const nativeAddon = resolve(PROJECT_ROOT, `packages/daemon-management/dist/native/${nativePlatform}-${info.arch}/desktop-host.node`)
  await access(nativeAddon)
  const binary = resolve(
    PROJECT_ROOT,
    "bin",
    `magnitude-cli${info.executableExt}`,
  )
  await mkdir(resolve(PROJECT_ROOT, "bin"), { recursive: true })
  if (info.platform === "darwin") {
    await runAppleBuild(compileAppleBun(resolve(PROJECT_ROOT, "cli/src/index.ts"), binary, target, "cli"))
    await runAppleBuild(signAppleCode(binary, "dev.magnitude.cli", "bun"))
    return binary
  }
  await run([
    process.execPath,
    "build",
    resolve(PROJECT_ROOT, "cli/src/index.ts"),
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
  ], { cwd: PROJECT_ROOT })
  return binary
}
