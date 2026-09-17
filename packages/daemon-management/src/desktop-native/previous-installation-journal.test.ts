import { mkdtemp, readFile, readdir, rm, stat, symlink, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { NodeContext } from "@effect/platform-node"
import { Effect, Layer, Option } from "effect"
import { describe, expect, it } from "vitest"
import { PreviousInstallationJournal, type PreviousInstallationPlan } from "./previous-installation"
import { previousInstallationJournal } from "./previous-installation-journal"
import { unixPrivateFilePermissions } from "./private-files"

const plan: PreviousInstallationPlan = { _tag: "Unix", startup: Option.none(), tree: Option.none() }
const fixture = async (test: (journal: PreviousInstallationJournal, directory: string) => Promise<void>) => {
  const directory = await mkdtemp(join(tmpdir(), "magnitude-upgrade-journal-"))
  try {
    const journal = await Effect.runPromise(PreviousInstallationJournal.pipe(Effect.provide(
      previousInstallationJournal(directory).pipe(Layer.provide(unixPrivateFilePermissions), Layer.provide(NodeContext.layer)))))
    await test(journal, directory)
  } finally { await rm(directory, { recursive: true, force: true }) }
}

describe.skipIf(process.platform === "win32")("previous installation recovery journal", () => {
  it("persists a private plan, replaces it atomically, and clears idempotently", async () => fixture(async (journal, directory) => {
    expect(Option.isNone(await Effect.runPromise(journal.read))).toBe(true)
    await Effect.runPromise(journal.write(plan))
    expect(Option.getOrThrow(await Effect.runPromise(journal.read))).toEqual(plan)
    expect((await stat(join(directory, "previous-installation.json"))).mode & 0o777).toBe(0o600)
    await Effect.runPromise(journal.write(plan))
    expect(await readdir(directory)).toEqual(["previous-installation.json"])
    await Effect.runPromise(journal.clear)
    await Effect.runPromise(journal.clear)
    expect(Option.isNone(await Effect.runPromise(journal.read))).toBe(true)
  }))
  it("rejects a malformed record instead of guessing at process identities", async () => fixture(async (journal, directory) => {
    await writeFile(join(directory, "previous-installation.json"), '{"_tag":"Unix","tree":{"pid":123}}')
    await expect(Effect.runPromise(journal.read)).rejects.toThrow("recovery record")
  }))
  it("refuses a symlink and preserves its target", async () => fixture(async (journal, directory) => {
    const target = join(directory, "preserved")
    await writeFile(target, "user data")
    await symlink(target, join(directory, "previous-installation.json"))
    await expect(Effect.runPromise(journal.read)).rejects.toThrow("Unsafe recovery record")
    expect(await readFile(target, "utf8")).toBe("user data")
  }))
})
