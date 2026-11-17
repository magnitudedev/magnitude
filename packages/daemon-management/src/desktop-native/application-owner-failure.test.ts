import { Effect, Option } from "effect"
import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { acquireApplicationMaintenance } from "./application-owner"
import { DataFolderUnusable } from "./application-control"
import { NativeHost, NativeHostUnavailable } from "./index"

const refusing = NativeHost.of({
  requireInteractiveDesktop: Effect.void, isServiceSession: Effect.succeed(false), localAppDataDirectory: Effect.succeed(""),
  acquireOwnership: () => Effect.fail(new NativeHostUnavailable({ message: "Error: Unsafe or inaccessible private ownership directory or lock" })),
  guardParent: () => Effect.void, ownedEndpoint: () => Effect.succeed(""), inspectEndpoint: () => Effect.succeed(Option.none()),
})

describe("an ownership lock that can't be opened", () => {
  it("names the data folder, so startup explains it instead of telling people to reinstall", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-owner-"))
    const directory = join(root, "a".repeat(40), "application")
    try {
      const failure = await Effect.runPromise(acquireApplicationMaintenance(directory).pipe(
        Effect.flip, Effect.scoped, Effect.provideService(NativeHost, refusing)))
      expect(failure).toBeInstanceOf(DataFolderUnusable)
      if (!(failure instanceof DataFolderUnusable)) return
      expect(failure.reason).toContain(directory)
      expect(failure.reason).not.toContain("Unsafe or inaccessible")
      expect(failure.message).toContain(directory)
      expect(failure.message).toContain("Unsafe or inaccessible private ownership directory or lock")
      expect(failure.message).not.toContain("Error:")
    } finally { await rm(root, { recursive: true, force: true }) }
  })
})
