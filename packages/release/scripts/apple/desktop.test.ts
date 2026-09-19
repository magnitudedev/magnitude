import * as FileSystem from "@effect/platform/FileSystem"
import { NodeContext } from "@effect/platform-node"
import { Effect } from "effect"
import { join } from "node:path"
import { expect, it } from "vitest"
import { attachDesktopDmg, detachDesktopDmg, packageDesktopDmg } from "./desktop"
import { appleCommand } from "./signing"

it.skipIf(process.platform !== "darwin").each(["HFS+", "APFS"])("ejects a %s disk image after its filesystem has already been unmounted", async filesystem => {
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const directory = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-dmg-detach-test-" })
    const image = join(directory, "test.dmg")
    const mount = join(directory, "mount")
    yield* fs.makeDirectory(mount)
    yield* appleCommand("/usr/bin/hdiutil", "create", "-size", "64m", "-fs", filesystem, "-volname", "MagnitudeDetachTest", image)
    yield* Effect.scoped(Effect.gen(function* () {
      const device = yield* Effect.acquireRelease(attachDesktopDmg(image, mount, "-readwrite"), device => detachDesktopDmg(device).pipe(Effect.orDie))
      expect(device).toMatch(/^\/dev\/disk\d+$/)
      // A busy eject can leave this state: the mount path is gone, but the device remains.
      yield* appleCommand("/usr/sbin/diskutil", "unmountDisk", device)
    }))
    expect(yield* appleCommand("/usr/bin/hdiutil", "info")).not.toContain(image)
  })).pipe(Effect.provide(NodeContext.layer)))
}, 30_000)

it.skipIf(process.platform !== "darwin")("persists Finder layout before sealing a fresh installer", async () => {
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const directory = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-dmg-layout-test-" })
    const app = join(directory, "Magnitude.app")
    yield* fs.makeDirectory(join(app, "Contents"), { recursive: true })
    yield* fs.writeFileString(join(app, "Contents/Info.plist"), '<?xml version="1.0"?><plist version="1.0"><dict><key>CFBundleName</key><string>Magnitude</string><key>CFBundlePackageType</key><string>APPL</string></dict></plist>')
    const image = join(directory, "installer.dmg"), mount = join(directory, "mounted")
    yield* packageDesktopDmg(app, image)
    yield* fs.makeDirectory(mount)
    yield* Effect.scoped(Effect.gen(function* () {
      yield* Effect.acquireRelease(attachDesktopDmg(image, mount, "-readonly"), device => detachDesktopDmg(device).pipe(Effect.orDie))
      expect(Number((yield* fs.stat(join(mount, ".DS_Store"))).size)).toBeGreaterThan(0)
      expect(yield* fs.readLink(join(mount, "Applications"))).toBe("/Applications")
      expect(yield* fs.exists(join(mount, ".background/background.tiff"))).toBe(true)
    }))
    expect(yield* appleCommand("/usr/bin/hdiutil", "info")).not.toContain(image)
  })).pipe(Effect.provide(NodeContext.layer)))
}, 60_000)
