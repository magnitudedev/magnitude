import { access, mkdir } from "node:fs/promises"
import { resolve } from "node:path"
import { getTargetInfo } from "../../../../scripts/release-target"
import { run } from "./common"
import { compileAppleBun, runAppleBuild } from "../apple/compile-bun"
import { signAppleCode } from "../apple/signing"
import { signWindowsCode } from "./windows-signing"
import { Effect } from "effect"
import { BunContext } from "@effect/platform-bun"
import { decodePublisherPublicKey } from "../../src/hosted-update/manifest"

const PROJECT_ROOT = resolve(import.meta.dir, "../../../..")

export const buildCliBinary = async (target: string, bootstrapPublisherKey?: string): Promise<string> => {
  if (bootstrapPublisherKey !== undefined) await Effect.runPromise(decodePublisherPublicKey(bootstrapPublisherKey))
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
    ...(bootstrapPublisherKey === undefined ? [] : ["--define", `MAGNITUDE_BOOTSTRAP_PUBLISHER_KEY=${JSON.stringify(bootstrapPublisherKey)}`]),
  ], { cwd: PROJECT_ROOT })
  if (info.platform === "windows") await Effect.runPromise(signWindowsCode(binary).pipe(Effect.provide(BunContext.layer)))
  return binary
}
