import { randomUUID } from "node:crypto"
import { Effect, Fiber, Ref, Stream } from "effect"
import { DesktopChildEvent, DesktopOwnerCommand } from "@magnitudedev/acn-protocol/desktop-control"
import { JsonLineChannelFailed, receiveJsonLines, sendJsonLine } from "@magnitudedev/utils/json-line-channel"
import { ProcessStartIdentitySchema } from "@magnitudedev/utils/process-groups"
import { OwnedChildSpawner, OwnedChildSpawnFailed, OwnedChildObservationFailed, OwnedChildRetirementFailed } from "./owned-child"
import { WindowsJobOwner } from "@magnitudedev/utils/windows-native"
import { WindowsPipeName, WindowsPrivatePipes } from "@magnitudedev/utils/windows-native"
import { encodeWindowsCommand } from "@magnitudedev/utils/windows-native"
import { windowsPipeDuplex } from "./windows-control"

/** The application-scoped job owner, not the attempt scope, retains native cleanup authority. */
export const makeWindowsOwnedChildSpawner = Effect.gen(function* () {
  const jobs = yield* WindowsJobOwner
  const pipes = yield* WindowsPrivatePipes
  return OwnedChildSpawner.of({ spawn: command => Effect.gen(function* () {
    const spawnFailure = (message: string) => new OwnedChildSpawnFailed({ executable: command.executable, message })
    const outputName = yield* Effect.sync(() => WindowsPipeName.make(`\\\\.\\pipe\\magnitude-output-${randomUUID()}`))
    const controlName = yield* Effect.sync(() => WindowsPipeName.make(`\\\\.\\pipe\\magnitude-child-${randomUUID()}`))
    const output = yield* pipes.bind(outputName, true).pipe(Effect.mapError(error => spawnFailure(error.message)))
    const control = yield* pipes.bind(controlName, true).pipe(Effect.mapError(error => spawnFailure(error.message)))
    const tail = yield* Ref.make("")
    const diagnosticReader = yield* output.accept.pipe(Effect.zipRight(Stream.repeatEffect(output.read).pipe(
      Stream.takeWhile(bytes => bytes.length > 0),
      Stream.runForEach(chunk => Ref.update(tail, text => Buffer.concat([Buffer.from(text), chunk]).subarray(-16384).toString("utf8"))),
    )), Effect.forkScoped)
    const encoded = yield* encodeWindowsCommand({ ...command, environment: {
      ...command.environment, MAGNITUDE_OWNER_PIPE: controlName,
    } }).pipe(Effect.mapError(error => spawnFailure(error.message)))
    const job = yield* Effect.acquireRelease(jobs.spawn(encoded, { _tag: "Diagnostics", output: outputName }).pipe(Effect.mapError(error => spawnFailure(error.message))),
      job => job.retire("10 seconds").pipe(Effect.catchAll(error => Effect.logError("Windows child cleanup remains unproven; application retains its job", error))),
    )
    const observed = yield* job.identity.pipe(Effect.mapError(error => spawnFailure(error.message)))
    const peer = yield* Effect.raceFirst(control.accept.pipe(Effect.mapError(error => spawnFailure(error.message))), job.exit.pipe(
      Effect.mapError(error => spawnFailure(error.message)), Effect.flatMap(code => Effect.fail(spawnFailure(`Magnitude service exited before connecting its control channel (exit code ${code}).`))),
    )).pipe(Effect.timeoutFail({ duration: "30 seconds", onTimeout: () => spawnFailure("Magnitude service did not connect its owned control channel.") }))
    if (peer !== observed.pid) return yield* spawnFailure("The Windows control connection does not belong to the owned service process.")
    const channel = yield* windowsPipeDuplex(control)
    return {
      identity: { pid: observed.pid, processStartIdentity: ProcessStartIdentitySchema.make(observed.creationTime) },
      stop: job.retire("10 seconds").pipe(Effect.mapError(error => new OwnedChildRetirementFailed({ pid: observed.pid, message: error.message }))),
      exit: job.exit.pipe(Effect.mapError(error => new OwnedChildObservationFailed({ pid: observed.pid, message: error.message }))),
      diagnosticTail: Ref.get(tail),
      events: Stream.merge(receiveJsonLines(channel, DesktopChildEvent), Stream.fromEffect(Fiber.join(diagnosticReader)).pipe(
        Stream.drain, Stream.mapError(error => new JsonLineChannelFailed({ message: error.message })),
      )),
      send: command => sendJsonLine(channel, DesktopOwnerCommand, command),
    }
  }) })
})
