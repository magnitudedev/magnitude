import { afterEach, expect, it } from "vitest"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { dirname, join } from "node:path"
import { hostById, inferenceNvrtcPaths, inferenceRequiredPaths, ReleaseAcquisitionError } from "@magnitudedev/release"
import { acquisitionError, isCompleteArtifact } from "./release-installation"

let directory: string | undefined

afterEach(async () => {
  if (directory) await rm(directory, { recursive: true, force: true })
  directory = undefined
})

it("says plainly that the inference engine couldn't be downloaded, from where, and what to do", () => {
  const explain = acquisitionError("https://releases.magnitude.dev/engine")
  expect(explain(new ReleaseAcquisitionError({ stage: "download", message: "release endpoint returned HTTP 404", transient: false })).message)
    .toBe("Magnitude couldn't download its inference engine (HTTP 404 from releases.magnitude.dev). Check your internet connection and try again.")
  expect(explain(new ReleaseAcquisitionError({ stage: "download", message: "release request failed", transient: true })).message)
    .toBe("Magnitude couldn't download its inference engine (request failed from releases.magnitude.dev). Check your internet connection and try again.")
  expect(explain(new ReleaseAcquisitionError({ stage: "verify", message: "artifact digest mismatch", transient: false })).message)
    .toBe("artifact digest mismatch")
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
