import * as FileSystem from "@effect/platform/FileSystem"
import { NodeContext } from "@effect/platform-node"
import { Effect, Exit, Fiber, Layer, TestClock, TestContext } from "effect"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { attachDesktopDmg, detachDesktopDmg, retryHdiutilBusy } from "./desktop"
import { AppleDistributionFailed, appleCommand } from "./signing"

describe("hdiutil transient busy retry", () => {
  it("retries Resource busy and temporarily unavailable failures and removes partial output between attempts", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const directory = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-hdiutil-retry-test-" })
      const output = join(directory, "installer.dmg")
      let attempts = 0
      const existedAtStart: boolean[] = []

      const value = yield* retryHdiutilBusy(
        Effect.gen(function* () {
          attempts += 1
          existedAtStart.push(yield* fs.exists(output).pipe(Effect.orDie))
          if (attempts === 1) {
            yield* fs.writeFileString(output, "partial-attempt-1").pipe(Effect.orDie)
            return yield* new AppleDistributionFailed({
              message: "/usr/bin/hdiutil exited 1: hdiutil: create failed - Resource busy",
            })
          }
          if (attempts === 2) {
            yield* fs.writeFileString(output, "partial-attempt-2").pipe(Effect.orDie)
            return yield* new AppleDistributionFailed({
              message: "/usr/bin/hdiutil exited 1: hdiutil: create failed - Resource temporarily unavailable",
            })
          }
          yield* fs.writeFileString(output, "complete-dmg").pipe(Effect.orDie)
          return "created"
        }),
        { cleanupOutput: output, baseDelay: "1 millis" },
      )

      return {
        value,
        attempts,
        existedAtStart,
        content: yield* fs.readFileString(output),
      }
    })).pipe(Effect.provide(NodeContext.layer)))

    expect(result).toEqual({
      value: "created",
      attempts: 3,
      existedAtStart: [false, false, false],
      content: "complete-dmg",
    })
  })

  it("uses exponential backoff and bounds repeated transient busy failures", async () => {
    const observed = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const directory = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-hdiutil-backoff-test-" })
      const output = join(directory, "installer.dmg")
      let attempts = 0

      const fiber = yield* Effect.fork(retryHdiutilBusy(
        Effect.gen(function* () {
          attempts += 1
          yield* fs.writeFileString(output, `partial-${attempts}`).pipe(Effect.orDie)
          return yield* new AppleDistributionFailed({
            message: "/usr/bin/hdiutil exited 1: hdiutil: create failed - Resource busy",
          })
        }),
        { cleanupOutput: output },
      ))

      const checkpoints: Array<{ readonly delay: string; readonly before: number; readonly after: number }> = []
      yield* TestClock.adjust("0 millis")
      for (const [beforeStep, finishStep, label] of [
        ["1999 millis", "1 millis", "2s"],
        ["3999 millis", "1 millis", "4s"],
        ["7999 millis", "1 millis", "8s"],
        ["15999 millis", "1 millis", "16s"],
        ["31999 millis", "1 millis", "32s"],
      ] as const) {
        yield* TestClock.adjust(beforeStep)
        const before = attempts
        yield* TestClock.adjust(finishStep)
        const after = attempts
        checkpoints.push({ delay: label, before, after })
      }

      const exit = yield* Fiber.await(fiber)
      const outputExists = yield* fs.exists(output)
      return { attempts, checkpoints, exit, outputExists }
    })).pipe(
      Effect.provide(Layer.merge(NodeContext.layer, TestContext.TestContext)),
    ))

    expect(observed.attempts).toBe(6)
    expect(observed.checkpoints).toEqual([
      { delay: "2s", before: 1, after: 2 },
      { delay: "4s", before: 2, after: 3 },
      { delay: "8s", before: 3, after: 4 },
      { delay: "16s", before: 4, after: 5 },
      { delay: "32s", before: 5, after: 6 },
    ])
    expect(Exit.isFailure(observed.exit)).toBe(true)
    expect(observed.outputExists).toBe(false)
  })

  it("does not retry non-transient hdiutil failures or delete an existing output file", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const directory = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-hdiutil-non-transient-test-" })
      const output = join(directory, "installer.dmg")
      yield* fs.writeFileString(output, "existing-dmg")
      let attempts = 0

      const exit = yield* Effect.exit(retryHdiutilBusy(
        Effect.gen(function* () {
          attempts += 1
          return yield* new AppleDistributionFailed({
            message: "/usr/bin/hdiutil exited 1: hdiutil: create failed - File exists",
          })
        }),
        { cleanupOutput: output, baseDelay: "1 millis" },
      ))

      return {
        attempts,
        failed: Exit.isFailure(exit),
        content: yield* fs.readFileString(output),
      }
    })).pipe(Effect.provide(NodeContext.layer)))

    expect(result).toEqual({
      attempts: 1,
      failed: true,
      content: "existing-dmg",
    })
  })
})

it.skipIf(process.platform !== "darwin").each(["HFS+", "APFS"])("ejects a %s disk image after its filesystem has already been unmounted", async filesystem => {
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const directory = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-dmg-detach-test-" })
    const image = join(directory, "test.dmg")
    const mount = join(directory, "mount")
    yield* fs.makeDirectory(mount)
    yield* retryHdiutilBusy(
      appleCommand("/usr/bin/hdiutil", "create", "-size", "64m", "-fs", filesystem, "-volname", "MagnitudeDetachTest", image),
      { cleanupOutput: image },
    )
    yield* Effect.scoped(Effect.gen(function* () {
      const device = yield* Effect.acquireRelease(attachDesktopDmg(image, mount, "-readwrite"), device => detachDesktopDmg(device).pipe(Effect.orDie))
      expect(device).toMatch(/^\/dev\/disk\d+$/)
      // A busy eject can leave this state: the mount path is gone, but the device remains.
      yield* appleCommand("/usr/sbin/diskutil", "unmountDisk", device)
    }))
    expect(yield* appleCommand("/usr/bin/hdiutil", "info")).not.toContain(image)
  })).pipe(Effect.provide(NodeContext.layer)))
}, 30_000)
