import { Deferred, Effect, Option, Runtime, Schema, Stream } from "effect"
import { createServer, type Socket } from "node:net"
import { dirname } from "node:path"
import { ApplicationIdentity } from "../application-identity"
import { AssertionFailure, InfrastructureFailure } from "../domain"
import { command } from "../process"
import { TerminalDriver } from "../terminal"

export const CliInterruption = Schema.Struct({ exitCode: Schema.Int, interrupted: Schema.Literal(true), stdout: Schema.String, stderr: Schema.String })
export const CliInterruptionConfig = Schema.Struct({ executable: Schema.NonEmptyString,
  port: Schema.Int.pipe(Schema.between(1024, 65535)), environment: Schema.Record({ key: Schema.String, value: Schema.String }) })
const fail = (message: string) => new AssertionFailure({ message })
const infrastructure = (message: string) => new InfrastructureFailure({ operation: "cli-interruption", message })

/** Caller obtains this identity from its isolated, owned desktop. The service is always resumed. */
export const verifyCliInterruption = (config: typeof CliInterruptionConfig.Type, identity: ApplicationIdentity) => Effect.scoped(Effect.gen(function* () {
  if (process.platform === "win32") return yield* infrastructure("Windows console interruption requires its native console driver")
  const signalService = (signal: "SIGSTOP" | "SIGCONT") => Effect.try({ try: () => process.kill(identity.servicePid, signal),
    catch: () => infrastructure(`Could not ${signal === "SIGSTOP" ? "pause" : "resume"} the owned test service`) })
  yield* Effect.acquireRelease(signalService("SIGSTOP"), () => signalService("SIGCONT").pipe(Effect.orDie))
  const child = yield* Effect.acquireRelease(Effect.try({ try: () => Bun.spawn([config.executable, "service", "status"], {
    env: config.environment, stdin: "ignore", stdout: "pipe", stderr: "pipe", detached: true,
  }), catch: () => infrastructure("Could not launch bundled CLI interruption probe") }), child => Effect.gen(function* () {
    if (child.exitCode === null) yield* Effect.sync(() => {
      try { process.kill(-child.pid, "SIGKILL") } catch (error) {
        if (!(error instanceof Error && "code" in error && error.code === "ESRCH")) throw error
      }
    })
    yield* Effect.promise(() => child.exited).pipe(Effect.interruptible, Effect.timeout("5 seconds"), Effect.orDie)
  }))
  const collect = (stream: ReadableStream<Uint8Array>) => Stream.fromReadableStream(() => stream, () => infrastructure("Cannot read interrupted CLI output")).pipe(
    Stream.runFoldEffect({ bytes: 0, chunks: [] as Uint8Array[] }, (state, chunk) => state.bytes + chunk.byteLength > 1024 * 1024
      ? Effect.fail(infrastructure("Interrupted CLI exceeded its output limit"))
      : Effect.succeed({ bytes: state.bytes + chunk.byteLength, chunks: [...state.chunks, chunk] })),
    Effect.map(value => Buffer.concat(value.chunks).toString("utf8")),
  )
  const interrupt = Effect.gen(function* () {
    for (;;) {
      if (child.exitCode !== null) return yield* fail("CLI exited before reaching the stalled service; interruption was not exercised")
      const sockets = yield* command("lsof", ["-nP", "-a", "-p", String(child.pid), `-iTCP@127.0.0.1:${config.port}`, "-sTCP:ESTABLISHED", "-Fn"],
        { timeoutMs: 5000, inheritEnv: false, env: config.environment })
      if (sockets.exitCode === 0 && sockets.stdout.split("\n").some(line => line.startsWith("n") && line.includes(`->127.0.0.1:${config.port}`))) break
      if (sockets.exitCode !== 0 && sockets.exitCode !== 1) return yield* infrastructure("Cannot observe the CLI connection to the owned service")
      yield* Effect.sleep("50 millis")
    }
    yield* Effect.try({ try: () => process.kill(-child.pid, "SIGINT"), catch: () => fail("CLI exited before its interrupt could be delivered") })
    const exitCode = yield* Effect.promise(() => child.exited).pipe(Effect.timeoutFail({ duration: "10 seconds", onTimeout: () => fail("CLI ignored interruption while waiting for the service") }))
    if (exitCode !== 130 && child.signalCode !== "SIGINT") return yield* fail(`Interrupted CLI returned unexpected exit code ${exitCode}`)
    return exitCode
  }).pipe(Effect.timeoutFail({ duration: "20 seconds", onTimeout: () => fail("CLI never connected to the deliberately stalled service") }))
  const [exitCode, stdout, stderr] = yield* Effect.all([interrupt, collect(child.stdout), collect(child.stderr)], { concurrency: "unbounded" })
  return CliInterruption.make({ exitCode, interrupted: true, stdout, stderr })
}))

/** A disposable silent endpoint stalls the installed CLI without suspending the Windows app owner. */
export const verifyWindowsCliInterruption = (config: typeof CliInterruptionConfig.Type & { readonly runtime: string; readonly evidence: string },
  onCleanupError: (message: string) => void) =>
  Effect.scoped(Effect.gen(function* () {
    const connected = yield* Deferred.make<void>()
    const runtime = yield* Effect.runtime<never>()
    const sockets = new Set<Socket>()
    const server = yield* Effect.acquireRelease(Effect.tryPromise({
      try: () => new Promise<ReturnType<typeof createServer>>((resolve, reject) => {
        const listener = createServer(socket => {
          sockets.add(socket)
          socket.on("close", () => sockets.delete(socket))
          Runtime.runSync(runtime)(Deferred.succeed(connected, undefined))
        })
        listener.once("error", reject)
        listener.listen(0, "127.0.0.1", () => { listener.off("error", reject); resolve(listener) })
      }),
      catch: () => infrastructure("Could not bind the isolated CLI interruption endpoint"),
    }), listener => Effect.tryPromise({
      try: () => new Promise<void>((resolve, reject) => {
        for (const socket of sockets) socket.destroy()
        listener.close(failure => failure ? reject(failure) : resolve())
      }),
      catch: () => infrastructure("Could not release the isolated CLI interruption endpoint"),
    }).pipe(Effect.catchAll(failure => Effect.sync(() => onCleanupError(failure.message)))))
    const address = server.address()
    if (!address || typeof address === "string") return yield* infrastructure("Isolated CLI endpoint has no TCP port")
    const terminal = yield* TerminalDriver
    const session = yield* terminal.start({ executable: config.executable, args: ["service", "status"],
      runtime: config.runtime, cwd: dirname(config.executable), evidence: config.evidence,
      environment: { ...config.environment, MAGNITUDE_DEV_PORT: String(address.port) }, columns: 80, rows: 24 },
    onCleanupError).pipe(Effect.mapError(() => infrastructure("Could not start the bundled CLI in its native console")))
    yield* Deferred.await(connected).pipe(Effect.timeoutFail({ duration: "15 seconds",
      onTimeout: () => fail("CLI did not connect to the isolated stalled endpoint") }))
    yield* session.write("\u0003")
    const result = yield* session.exited.pipe(Effect.timeoutFail({ duration: "10 seconds",
      onTimeout: () => fail("CLI ignored terminal interruption while waiting for the endpoint") }))
    if (![130, -1073741510, 3221225786].includes(result.code) && !Option.contains(result.signal, "SIGINT")) {
      return yield* fail(`CLI returned unexpected Windows interruption exit code ${result.code}`)
    }
    const screen = yield* session.screen
    return CliInterruption.make({ exitCode: result.code, interrupted: true, stdout: screen.lines.join("\n"), stderr: "" })
  }))
