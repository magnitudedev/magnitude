import { afterEach, expect, it } from "vitest"
import { BunContext } from "@effect/platform-bun"
import { Effect, Option } from "effect"
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { dirname, join } from "node:path"
import { hostById, inferenceNvrtcPaths, inferenceRequiredPaths, ReleaseAcquisitionError, ReleaseDiskFullError } from "@magnitudedev/release"
import { acquisitionError, diskFullError, isCompleteArtifact } from "./release-installation"

let directory: string | undefined

afterEach(async () => {
  if (directory) await rm(directory, { recursive: true, force: true })
  directory = undefined
})

it("says plainly that the inference engine couldn't be downloaded, from where, and what to do about the cause", () => {
  const explain = acquisitionError("https://releases.magnitude.dev/engine")
  expect(explain(new ReleaseAcquisitionError({ stage: "download", message: "release endpoint returned HTTP 404", transient: false })).message)
    .toBe("Magnitude couldn't download its inference engine (HTTP 404 from releases.magnitude.dev). This release isn't available for download. Update Magnitude, or try again later.")
  expect(explain(new ReleaseAcquisitionError({ stage: "download", message: "release endpoint returned HTTP 403", transient: false })).message)
    .toBe("Magnitude couldn't download its inference engine (HTTP 403 from releases.magnitude.dev). The release host refused the download. Try again later.")
  expect(explain(new ReleaseAcquisitionError({ stage: "download", message: "release request failed", transient: true })).message)
    .toBe("Magnitude couldn't download its inference engine (request failed from releases.magnitude.dev). Check your internet connection and try again.")
  expect(explain(new ReleaseAcquisitionError({ stage: "download", message: "release endpoint returned HTTP 404", transient: false })).permanent).toBe(true)
  expect(explain(new ReleaseAcquisitionError({ stage: "download", message: "release request failed", transient: true })).permanent).toBe(false)
  expect(explain(new ReleaseAcquisitionError({ stage: "verify", message: "artifact digest mismatch", transient: false })).message)
    .toBe("artifact digest mismatch")
})

it("says the disk is full, where, and how much space the engine needs, as a failure a restart cannot fix", () => {
  const explain = diskFullError("/home/qa/.magnitude")
  const measured = explain(new ReleaseDiskFullError({
    directory: "/home/qa/.magnitude/releases/artifacts/0.2.9/linux-x64-gnu",
    requiredBytes: Option.some(1_449_725_607),
    availableBytes: Option.some(612_000_000),
  }))
  expect(measured.message).toBe("There isn't enough free disk space in /home/qa/.magnitude to install Magnitude's inference engine. "
    + "It needs about 1.5 GB free, and 612 MB is free. Free up space on that disk, then start Magnitude again.")
  expect(measured.permanent).toBe(true)
  expect(explain(new ReleaseDiskFullError({ directory: "/home/qa/.magnitude", requiredBytes: Option.none(), availableBytes: Option.none() })).message)
    .toBe("There isn't enough free disk space in /home/qa/.magnitude to install Magnitude's inference engine. Free up space on that disk, then start Magnitude again.")
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
