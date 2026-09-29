import { afterEach, expect, it } from "vitest"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { dirname, join } from "node:path"
import { hostById, inferenceNvrtcPaths, inferenceRequiredPaths } from "@magnitudedev/release"
import { isCompleteArtifact } from "./release-installation"

let directory: string | undefined

afterEach(async () => {
  if (directory) await rm(directory, { recursive: true, force: true })
  directory = undefined
})

it("requires every bundled NVRTC file before reusing a cached Linux installation", async () => {
  directory = await mkdtemp(join(tmpdir(), "magnitude-inference-installation-"))
  const host = hostById("linux-x64-gnu")
  const paths = inferenceRequiredPaths(host)
  for (const relative of paths) {
    const file = join(directory, relative)
    await mkdir(dirname(file), { recursive: true })
    await writeFile(file, "present")
  }
  const complete = () => Effect.runPromise(
    isCompleteArtifact(directory!, host.id).pipe(Effect.provide(BunContext.layer)),
  )
  expect(await complete()).toBe(true)
  await rm(join(directory, inferenceNvrtcPaths(host)[0]!))
  expect(await complete()).toBe(false)
})
