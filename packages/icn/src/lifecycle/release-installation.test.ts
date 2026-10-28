import { afterEach, expect, it } from "vitest"
import { BunContext } from "@effect/platform-bun"
import { Effect, Fiber, Ref, TestClock, TestContext } from "effect"
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { dirname, join } from "node:path"
import { hostById, inferenceNvrtcPaths, inferenceRequiredPaths, ReleaseAcquisitionError } from "@magnitudedev/release"
import { IcnPreparationReporter, type IcnPreparationEvent } from "./preparation"
import { acquisitionError, isCompleteArtifact, ReleaseIcnInstallationError, retryAcquisition } from "./release-installation"

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
    .toBe("Magnitude couldn't download its inference engine (request failed from releases.magnitude.dev). Check your internet connection.")
  expect(explain(new ReleaseAcquisitionError({ stage: "download", message: "release endpoint returned HTTP 404", transient: false })).permanent).toBe(true)
  expect(explain(new ReleaseAcquisitionError({ stage: "download", message: "release request failed", transient: true })).permanent).toBe(false)
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

it("acquires the engine again after a failure starting again can fix, with growing waits it reports", async () => {
  const events: IcnPreparationEvent[] = []
  const reporter = { report: (event: IcnPreparationEvent) => Effect.sync(() => { events.push(event) }) }
  const result = await Effect.runPromise(Effect.gen(function* () {
    const attempts = yield* Ref.make(0)
    const acquisition = Ref.updateAndGet(attempts, n => n + 1).pipe(Effect.flatMap(n => n < 4
      ? Effect.fail(new ReleaseIcnInstallationError({ stage: "acquire", message: `outage ${n}`, permanent: false }))
      : Effect.succeed("installed")))
    const fiber = yield* Effect.fork(retryAcquisition(acquisition))
    yield* TestClock.adjust("5 seconds")
    yield* TestClock.adjust("10 seconds")
    yield* TestClock.adjust("20 seconds")
    return { value: yield* Fiber.join(fiber), attempts: yield* Ref.get(attempts) }
  }).pipe(Effect.provideService(IcnPreparationReporter, reporter), Effect.provide(TestContext.TestContext)))
  expect(result).toEqual({ value: "installed", attempts: 4 })
  expect(events).toEqual([
    { _tag: "Retrying", message: "outage 1", retryAtMillis: 5_000 },
    { _tag: "Retrying", message: "outage 2", retryAtMillis: 15_000 },
    { _tag: "Retrying", message: "outage 3", retryAtMillis: 35_000 },
  ])
})

it("ends startup on a failure starting again can't fix", async () => {
  const events: IcnPreparationEvent[] = []
  const reporter = { report: (event: IcnPreparationEvent) => Effect.sync(() => { events.push(event) }) }
  const error = await Effect.runPromise(retryAcquisition(
    Effect.fail(new ReleaseIcnInstallationError({ stage: "acquire", message: "HTTP 404", permanent: true })),
  ).pipe(Effect.flip, Effect.provideService(IcnPreparationReporter, reporter)))
  expect(error.message).toBe("HTTP 404")
  expect(events).toEqual([])
})
