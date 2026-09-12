import { randomUUID } from "node:crypto"
import { Effect, Layer, Schema, Stream } from "effect"
import { IcnChildSpawner, IcnChildLaunch, IcnChildAcquisitionFailed, IcnChildObservationFailed, IcnChildRetirementFailed } from "@magnitudedev/icn"
import { IcnParentCommand } from "@magnitudedev/icn-protocol"
import { WindowsJobOwner, WindowsPrivatePipes, WindowsPipeName, encodeWindowsCommand, type WindowsPrivatePipe } from "@magnitudedev/utils/windows-native"

/** ACN's job scope outlives the attempt scope and retains failed-retirement authority. */
export const WindowsIcnChildSpawner = Layer.effect(IcnChildSpawner, Effect.gen(function* () {
  const jobs = yield* WindowsJobOwner
  const pipes = yield* WindowsPrivatePipes
  return IcnChildSpawner.of({ spawn: input => Effect.gen(function* () {
    const launch = yield* Schema.validate(IcnChildLaunch)(input)
    const acquisition = (message: string) => new IcnChildAcquisitionFailed({ executable: launch.executable, message })
    const names = yield* Effect.sync(() => {
      const id = randomUUID()
      return { input: WindowsPipeName.make(`\\\\.\\pipe\\magnitude-icn-${id}-stdin`), output: WindowsPipeName.make(`\\\\.\\pipe\\magnitude-icn-${id}-stdout`), error: WindowsPipeName.make(`\\\\.\\pipe\\magnitude-icn-${id}-stderr`) }
    })
    const stdin = yield* pipes.bind(names.input, true).pipe(Effect.mapError(error => acquisition(error.message)))
    const stdout = yield* pipes.bind(names.output, true).pipe(Effect.mapError(error => acquisition(error.message)))
    const stderr = yield* pipes.bind(names.error, true).pipe(Effect.mapError(error => acquisition(error.message)))
    const command = yield* encodeWindowsCommand({ executable: launch.executable, arguments: launch.arguments, environment: { ...process.env, ...launch.environment } }).pipe(Effect.mapError(error => acquisition(error.message)))
    const shutdownFrame = yield* Schema.encode(Schema.parseJson(IcnParentCommand))({ type: "shutdown" })
    const owned = yield* Effect.acquireRelease(Effect.gen(function* () {
      const job = yield* jobs.spawn(command, { _tag: "Separate", ...names }).pipe(Effect.mapError(error => acquisition(error.message)))
      const shutdown = yield* Effect.cached(Effect.gen(function* () {
        yield* stdin.write(Buffer.from(`${shutdownFrame}\n`)).pipe(Effect.zipRight(job.exit),
          Effect.interruptible, Effect.timeoutOption(launch.gracefulShutdownTimeout),
          Effect.catchAll(() => Effect.void))
        // Root exit does not prove nested workers retired. Always finish exact job retirement.
        yield* job.retire(launch.forceShutdownTimeout).pipe(Effect.interruptible)
      }).pipe(Effect.uninterruptible))
      return { job, shutdown }
    }), owned => owned.shutdown.pipe(Effect.catchAll(error => Effect.logError("ICN cleanup remains unproven; ACN retains its Windows job", error))))
    const { job } = owned
    const identity = yield* job.identity.pipe(Effect.mapError(error => acquisition(error.message)))
    yield* Effect.raceFirst(
      Effect.all([stdin.accept, stdout.accept, stderr.accept], { concurrency: "unbounded" }).pipe(Effect.mapError(error => acquisition(error.message))),
      job.exit.pipe(Effect.mapError(error => acquisition(error.message)), Effect.flatMap(code => Effect.fail(acquisition(`Inference exited before its inherited streams were admitted (exit code ${code}).`)))),
    ).pipe(Effect.timeoutFail({ duration: "10 seconds", onTimeout: () => acquisition("Inference inherited-stream admission timed out.") }))
    const observation = (message: string) => new IcnChildObservationFailed({ pid: identity.pid, message })
    const read = (pipe: WindowsPrivatePipe) => Stream.repeatEffect(pipe.read).pipe(Stream.takeWhile(bytes => bytes.length > 0), Stream.mapError(error => observation(error.message)))
    return {
      pid: identity.pid, stdout: read(stdout), stderr: read(stderr),
      exitCode: job.exit.pipe(Effect.mapError(error => observation(error.message))),
      terminate: owned.shutdown.pipe(Effect.mapError(error => new IcnChildRetirementFailed({ pid: identity.pid, message: error.message }))),
    }
  }) })
}))
