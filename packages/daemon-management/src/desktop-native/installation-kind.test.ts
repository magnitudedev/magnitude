import { BunContext } from "@effect/platform-bun"
import { Effect, Option } from "effect"
import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { isDockerInstallation, readLinuxInstallationKind } from "./installation-kind"
import { headlessCapabilities } from "./headless-application"

describe("installation kind", () => {
  it("reads the marker the package or image writes, and treats an unreadable one as no kind", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-installation-kind-"))
    const read = () => Effect.runPromise(Effect.all([readLinuxInstallationKind(root), isDockerInstallation(root)]).pipe(Effect.provide(BunContext.layer)))
    try {
      expect(await read()).toEqual([Option.none(), false])
      for (const [marker, expected] of [['{"format":"docker"}', [Option.some("docker"), true]], ['{"format":"deb"}', [Option.some("deb"), false]],
        ['{"format":"snap"}', [Option.none(), false]], ["not json", [Option.none(), false]]] as const) {
        await writeFile(join(root, "update-package.json"), marker)
        expect(await read()).toEqual(expected)
      }
    } finally { await rm(root, { recursive: true, force: true }) }
  })
  it("gives a container's headless owner only restarting", () => {
    expect(headlessCapabilities(true)).toEqual(["RestartService"])
    expect(headlessCapabilities(false)).toEqual(["Updates", "RestartService", "Quit"])
  })
})
