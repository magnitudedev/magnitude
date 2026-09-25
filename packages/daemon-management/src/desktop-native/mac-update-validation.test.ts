import { Command, CommandExecutor, FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Scope } from "effect"
import { createHash, generateKeyPairSync } from "node:crypto"
import { createRequire } from "node:module"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"
import { MacBundleVerificationFailed, MacBundleVerifier, nativeMacBundleVerifier } from "./mac-update-validation"
import { MacUpdateFilesystem, nativeMacUpdateFilesystem } from "./mac-update-filesystem"
import { exchangeMacUpdate } from "./mac-update-transaction"
import { retireMacUpdateBundle } from "./mac-update-recovery"
import { makeMacUpdateArchiveStager } from "./mac-update-staging"
import { guardedCommandLayer } from "./guarded-command"
import { signUpdateRelease } from "../../../release/src/hosted-update/release"

const addon = fileURLToPath(new URL(`../../dist/native/darwin-${process.arch}/desktop-host.node`, import.meta.url))
const rule = 'identifier "dev.magnitude.desktop"'
const architecture = process.arch === "arm64" ? "arm64" : "x86_64"
const command = (...args: [string, ...string[]]) => Command.make(...args).pipe(Command.exitCode,
  Effect.tap(code => Effect.sync(() => expect(code).toBe(0))))
const plist = (identifier: string) => `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict><key>CFBundleIdentifier</key><string>${identifier}</string>
<key>CFBundleExecutable</key><string>program</string><key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>0.1.6</string><key>CFBundleVersion</key><string>6</string></dict></plist>`

const fixture = <A, E>(use: (bundle: string, root: string) => Effect.Effect<A, E, FileSystem.FileSystem | CommandExecutor.CommandExecutor | Scope.Scope>) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-bundle-validation-" })
  const bundle = join(root, "Magnitude π.app")
  yield* fs.makeDirectory(join(bundle, "Contents/MacOS"), { recursive: true })
  yield* fs.makeDirectory(join(bundle, "Contents/Resources"))
  yield* fs.writeFileString(join(bundle, "Contents/Info.plist"), plist("dev.magnitude.desktop"))
  yield* fs.writeFileString(join(bundle, "Contents/Resources/data"), "sealed resource")
  yield* fs.writeFileString(join(root, "program.c"), "int main(void) { return 0; }\n")
  yield* command("/usr/bin/cc", "-mmacosx-version-min=13.0", join(root, "program.c"), "-o", join(bundle, "Contents/MacOS/program"))
  yield* command("/usr/bin/codesign", "--force", "--sign", "-", bundle)
  return yield* use(bundle, root)
}).pipe(Effect.scoped, Effect.provide(BunContext.layer), Effect.runPromise)

const verify = (bundle: string, requirement = rule, version = "0.1.6", arch = architecture) => Effect.tryPromise(() => {
  const native = createRequire(import.meta.url)(addon) as { verifyMacBundle: (path: string, requirement: string, version: string, architecture: string) => Promise<void> }
  return native.verifyMacBundle(bundle, requirement, version, arch)
})

describe.skipIf(process.platform !== "darwin")("native macOS bundle verification", () => {
  it("installs two successive signed archive fixtures with transaction recovery and displaced-bundle cleanup", () => fixture((bundle, root) => Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const native = yield* MacUpdateFilesystem
    const stagingPath = join(root, "transaction")
    yield* fs.makeDirectory(stagingPath, { mode: 0o700 })
    const payload = join(root, "payload/Magnitude.app")
    yield* fs.makeDirectory(join(root, "payload"))
    yield* fs.copy(bundle, payload)
    const archive = join(root, "update.zip")
    yield* command("/usr/bin/ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", payload, archive)
    const bytes = yield* fs.readFile(archive)
    const key = generateKeyPairSync("ed25519")
    const release = yield* signUpdateRelease({ version: "0.1.6", bytes: bytes.length,
      sha256: createHash("sha256").update(bytes).digest("hex") },
      { os: "darwin", arch: process.arch === "arm64" ? "arm64" : "x64", package: "mac-zip" }, key.privateKey)
    const stager = yield* makeMacUpdateArchiveStager({
      helper: join(addon, "../magnitude-extract"), architecture: process.arch === "arm64" ? "arm64" : "x64",
      trustedPublishers: new Map([["fixture", key.publicKey]]),
    })
    yield* fs.writeFileString(join(bundle, "Contents/Info.plist"), plist("dev.magnitude.desktop").replace("0.1.6", "0.1.5"))
    yield* command("/usr/bin/codesign", "--force", "--sign", "-", bundle)
    const installed = yield* native.open(root, false)
    const staging = yield* native.open(stagingPath, true)
    expect(yield* stager.stage(archive, staging, { ...release, sha256: "0".repeat(64) }).pipe(Effect.isFailure)).toBe(true)
    expect(yield* fs.readDirectory(stagingPath)).toEqual([])
    yield* fs.writeFile(archive, Buffer.concat([bytes, Buffer.from("changed")]))
    expect(yield* stager.stage(archive, staging, release).pipe(Effect.isFailure)).toBe(true)
    expect(yield* fs.readDirectory(stagingPath)).toEqual([])
    yield* fs.writeFile(archive, bytes)
    yield* stager.stage(archive, staging, release)
    const result = yield* exchangeMacUpdate(installed, "Magnitude π.app", staging, {
      previous: "0.1.5", replacement: "0.1.6", architecture: process.arch === "arm64" ? "arm64" : "x64",
    })
    expect(result).toEqual({ _tag: "Installed", version: "0.1.6" })
    yield* verify(bundle, rule, "0.1.6")
    yield* verify(join(stagingPath, "Magnitude.app"), rule, "0.1.5")
    expect(yield* retireMacUpdateBundle(installed, "Magnitude π.app", staging)).toEqual(result)
    expect(yield* fs.exists(join(stagingPath, "Magnitude.app"))).toBe(false)
    yield* verify(bundle, rule, "0.1.6")

    yield* fs.writeFileString(join(payload, "Contents/Info.plist"), plist("dev.magnitude.desktop").replace("0.1.6", "0.1.7"))
    yield* command("/usr/bin/codesign", "--force", "--sign", "-", payload)
    yield* command("/usr/bin/ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", payload, archive)
    const nextBytes = yield* fs.readFile(archive)
    const nextRelease = yield* signUpdateRelease({ version: "0.1.7", bytes: nextBytes.length,
      sha256: createHash("sha256").update(nextBytes).digest("hex") },
      { os: "darwin", arch: process.arch === "arm64" ? "arm64" : "x64", package: "mac-zip" }, key.privateKey)
    const nextPath = join(root, "next-transaction")
    yield* fs.makeDirectory(nextPath, { mode: 0o700 })
    const next = yield* native.open(nextPath, true)
    yield* stager.stage(archive, next, nextRelease)
    expect(yield* exchangeMacUpdate(installed, "Magnitude π.app", next, {
      previous: "0.1.6", replacement: "0.1.7", architecture: process.arch === "arm64" ? "arm64" : "x64",
    })).toEqual({ _tag: "Installed", version: "0.1.7" })
    yield* verify(bundle, rule, "0.1.7")
    yield* retireMacUpdateBundle(installed, "Magnitude π.app", next)
    yield* verify(bundle, rule, "0.1.7")
    expect(yield* fs.readDirectory(stagingPath)).toEqual([])
    expect(yield* fs.readDirectory(nextPath)).toEqual([])
  }).pipe(Effect.provide([nativeMacUpdateFilesystem(addon), guardedCommandLayer(join(addon, "../magnitude-command"))]), Effect.provideService(MacBundleVerifier, {
    verify: (path, expected) => verify(path, rule, expected.version, expected.architecture === "x64" ? "x86_64" : "arm64").pipe(
      Effect.mapError(() => new MacBundleVerificationFailed())),
  }))))

  it("validates sealed local fixtures without granting production publisher trust", () => fixture(bundle => Effect.gen(function* () {
    yield* verify(bundle)
    expect(yield* verify(bundle, `${rule} and anchor apple generic`).pipe(Effect.isFailure)).toBe(true)
  })))

  it.each([
    ["bundle identifier", 'identifier "dev.magnitude.other"', "0.1.6", architecture],
    ["version", rule, "0.1.7", architecture],
    ["architecture", rule, "0.1.6", architecture === "arm64" ? "x86_64" : "arm64"],
    ["invalid requirement", "not a valid requirement (", "0.1.6", architecture],
  ])("rejects an unexpected %s", (_name, requirement, version, arch) => fixture(bundle => Effect.gen(function* () {
    expect(yield* verify(bundle, requirement, version, arch).pipe(Effect.isFailure)).toBe(true)
  })))

  it("rejects changed sealed resources", () => fixture(bundle => Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    yield* fs.writeFileString(join(bundle, "Contents/Resources/data"), "modified")
    expect(yield* verify(bundle).pipe(Effect.isFailure)).toBe(true)
  })))

  it("accepts a sealed framework with version links and rejects changed nested code", () => fixture((bundle, root) => Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const framework = join(bundle, "Contents/Frameworks/Fixture.framework")
    const version = join(framework, "Versions/A")
    yield* fs.makeDirectory(join(version, "Resources"), { recursive: true })
    yield* fs.writeFileString(join(version, "Resources/Info.plist"), plist("dev.magnitude.fixture")
      .replace("<string>program</string>", "<string>Fixture</string>").replace("<string>APPL</string>", "<string>FMWK</string>"))
    yield* command("/usr/bin/cc", "-dynamiclib", "-mmacosx-version-min=13.0", join(root, "program.c"), "-o", join(version, "Fixture"))
    yield* fs.symlink("A", join(framework, "Versions/Current"))
    yield* fs.symlink("Versions/Current/Fixture", join(framework, "Fixture"))
    yield* fs.symlink("Versions/Current/Resources", join(framework, "Resources"))
    yield* command("/usr/bin/codesign", "--force", "--sign", "-", framework)
    yield* command("/usr/bin/codesign", "--force", "--sign", "-", bundle)
    yield* verify(bundle)
    yield* command("/usr/bin/codesign", "--remove-signature", join(version, "Fixture"))
    expect(yield* verify(bundle).pipe(Effect.isFailure)).toBe(true)
  })))

  it("rejects an identifier that disagrees with the sealed bundle metadata", () => fixture(bundle => Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    yield* fs.writeFileString(join(bundle, "Contents/Info.plist"), plist("dev.magnitude.other"))
    yield* command("/usr/bin/codesign", "--force", "--sign", "-", "--identifier", "dev.magnitude.desktop", bundle)
    expect(yield* verify(bundle).pipe(Effect.isFailure)).toBe(true)
  })))

  it("checks the non-native slice of a universal executable", () => fixture((bundle, root) => Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const otherBundle = join(root, "Other.app")
    const program = join(bundle, "Contents/MacOS/program")
    const otherProgram = join(otherBundle, "Contents/MacOS/program")
    const nativeSlice = join(root, "native-slice")
    yield* fs.copy(bundle, otherBundle)
    yield* fs.copyFile(program, nativeSlice)
    yield* fs.remove(otherProgram)
    yield* command("/usr/bin/cc", "-arch", architecture === "arm64" ? "x86_64" : "arm64", "-mmacosx-version-min=13.0",
      join(root, "program.c"), "-o", otherProgram)
    yield* command("/usr/bin/codesign", "--force", "--sign", "-", otherBundle)
    yield* command("/usr/bin/lipo", "-create", nativeSlice, otherProgram, "-output", program)
    yield* verify(bundle)
    yield* command("/usr/bin/codesign", "--force", "--sign", "-", "--identifier", "dev.magnitude.other", otherBundle)
    yield* command("/usr/bin/lipo", "-create", nativeSlice, otherProgram, "-output", program)
    expect(yield* verify(bundle).pipe(Effect.isFailure)).toBe(true)
  })))

  it("rejects unsigned code and leaf symlinks", () => fixture((bundle, root) => Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const link = join(root, "Linked.app")
    yield* fs.symlink(bundle, link)
    expect(yield* verify(link).pipe(Effect.isFailure)).toBe(true)
    yield* command("/usr/bin/codesign", "--remove-signature", join(bundle, "Contents/MacOS/program"))
    expect(yield* verify(bundle).pipe(Effect.isFailure)).toBe(true)
  })))

  it("rejects embedded NUL instead of verifying a truncated path", () => fixture(bundle => Effect.gen(function* () {
    expect(yield* verify(`${bundle}\0ignored`).pipe(Effect.isFailure)).toBe(true)
  })))
})

it.skipIf(process.platform !== "darwin")("does not construct production update trust without a compiled publisher", async () => {
  // An existing, loadable addon prevents missing bindings from satisfying this trust assertion.
  expect(typeof createRequire(import.meta.url)(addon).verifyMacBundle).toBe("function")
  const result = await Effect.runPromise(MacBundleVerifier.pipe(Effect.provide(nativeMacBundleVerifier(addon)), Effect.either))
  expect(result._tag).toBe("Left")
})
