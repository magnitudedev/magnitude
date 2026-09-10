import { BunContext } from "@effect/platform-bun"
import { ConfigProvider, Effect } from "effect"
import { createHash } from "node:crypto"
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { verifyAppleReceipts } from "./verify-receipts"
import { backendPacks, releaseHosts } from "../../src/targets"
const commit = "a".repeat(40), team = "ABCDEFGHIJ"
let root: string
const run = () => Effect.runPromise(verifyAppleReceipts(root).pipe(Effect.withConfigProvider(ConfigProvider.fromMap(new Map([["MAGNITUDE_SOURCE_COMMIT", commit], ["APPLE_TEAM_ID", team]]))), Effect.provide(BunContext.layer)))
const write = (path: string, value: unknown) => writeFile(path, JSON.stringify(value))
const notary = (unit: string) => ({ id: `accepted-${unit}`, unit, status: "Accepted", inputSha256: "b".repeat(64) })
beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "magnitude-apple-receipts-"))
  const groups = [
    ...releaseHosts.filter((host) => host.id.startsWith("darwin-")).map((host) => ({ directory: host.id, host: host.id, ids: ["cli", "acn", "icn-base"].map((kind) => ({ id: `${kind}-${host.id}`, kind })), units: ["cli", "inference", "app"], stapledApp: true })),
    ...backendPacks.filter((pack) => pack.host.startsWith("darwin-")).map((pack) => ({ directory: pack.id, host: pack.host, ids: [{ id: `icn-backend-${pack.id}`, kind: "icn-backend" }], units: [pack.id], stapledApp: false })),
  ]
  for (const group of groups) {
    const directory = join(root, group.directory)
    await mkdir(directory)
    const artifacts = []
    for (const { id, kind } of group.ids) {
      const sha256 = createHash("sha256").update(id).digest("hex")
      artifacts.push({ id, sha256 })
      await writeFile(join(directory, `${id}.tar.gz`), id)
      await write(join(directory, `${id}.artifact.json`), { id, kind, host: group.host, filename: `${id}.tar.gz`, bytes: id.length, sha256 })
    }
    await write(join(directory, "apple-distribution.receipt.json"), { sourceCommit: commit, team, artifacts, notarizations: group.units.map(notary), stapledApp: group.stapledApp })
    if (group.stapledApp) await write(join(directory, "apple-consumer.receipt.json"), { sourceCommit: commit, artifacts })
  }
})
afterEach(() => rm(root, { recursive: true, force: true }))
describe("Apple publication receipts", () => {
  it("accepts the complete notarized and independently consumed graph", async () => { await run() })
  it("rejects changed final archive bytes", async () => {
    await writeFile(join(root, "darwin-arm64/cli-darwin-arm64.tar.gz"), "changed")
    await expect(run()).rejects.toThrow("Apple accepted bytes changed")
  })
  it("rejects missing independent consumer validation", async () => {
    await rm(join(root, "darwin-arm64/apple-consumer.receipt.json"))
    await expect(run()).rejects.toThrow("independent Apple host consumer")
  })
  it("rejects an unstapled app", async () => {
    const file = join(root, "darwin-arm64/apple-distribution.receipt.json")
    const receipt = JSON.parse(await readFile(file, "utf8")); receipt.stapledApp = false
    await write(file, receipt)
    await expect(run()).rejects.toThrow("stapled ticket")
  })
  it("rejects another publisher", async () => {
    const file = join(root, "darwin-arm64/apple-distribution.receipt.json")
    const receipt = JSON.parse(await readFile(file, "utf8")); receipt.team = "ZYXWVUTSRQ"
    await write(file, receipt)
    await expect(run()).rejects.toThrow("selected commit and publisher")
  })
})
