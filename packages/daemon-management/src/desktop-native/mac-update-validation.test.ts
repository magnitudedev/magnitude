import { Command, CommandExecutor, FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { createRequire } from "node:module"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"
import { MacBundleVerifier, nativeMacBundleVerifier } from "./mac-update-validation"

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

const fixture = <A, E>(use: (bundle: string, root: string) => Effect.Effect<A, E, FileSystem.FileSystem | CommandExecutor.CommandExecutor>) => Effect.gen(function* () {
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
