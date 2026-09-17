import * as FileSystem from "@effect/platform/FileSystem"
import { NodeContext } from "@effect/platform-node"
import { Effect } from "effect"
import { join } from "node:path"
import { expect, it } from "vitest"
import { attachDesktopDmg, detachDesktopDmg } from "./desktop"
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
