import { cp, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises"
import { homedir, tmpdir } from "node:os"
import { join, resolve } from "node:path"
import { Data, Effect, Fiber, Option, Schema } from "effect"
import {
  HeadlessSessionIdSchema,
  type HeadlessSessionId,
} from "@magnitudedev/client-common"
import {
  attestAcnProcessTreeExit,
  waitForAcnProcessAttestation,
  type AcnProcessAttestation,
} from "@magnitudedev/sdk/testing"
import {
  validateDurableSessionMetadata,
  validateFakeInferenceRequest,
  validateFinalFakeInferenceState,
} from "./e2e-verification"

const cliDir = resolve(import.meta.dir, "..", "..")
const bun = process.execPath
const prompt = "Reply with exactly headless-ok."
const systemOverride = "Do not use tools. Reply with exactly headless-ok."
const inheritedEnvironmentNames = [
  "LANG",
  "LC_ALL",
  "LC_CTYPE",
  "LOGNAME",
  "PATH",
  "SHELL",
  "TEMP",
  "TMP",
  "TMPDIR",
  "USER",
] as const

const isolatedEnvironment = (home: string, icnPath: string): Record<string, string> => {
  const entries = inheritedEnvironmentNames.flatMap((name) => {
    const value = process.env[name]
    return value === undefined ? [] : [[name, value] as const]
  })
  const environment = Object.fromEntries(entries)
  return {
    ...environment,
    CI: "1",
    FORCE_COLOR: "0",
    HOME: home,
    MAGNITUDE_ICN_PATH: icnPath,
    NO_COLOR: "1",
    NO_PROXY: "127.0.0.1,localhost",
    TERM: "dumb",
  }
}

interface CommandResult {
  readonly exitCode: number
  readonly stdout: string
  readonly stderr: string
  readonly timedOut: boolean
}

class HarnessFailure extends Data.TaggedError("HarnessFailure")<{
  readonly operation: string
  readonly message: string
}> {}

const failure = (operation: string, cause: unknown): HarnessFailure => new HarnessFailure({
  operation,
  message: cause instanceof Error ? cause.message : String(cause),
})

const runCommand = (
  args: readonly string[],
  options: {
    readonly home: string
    readonly icnPath: string
    readonly label: string
    readonly timeoutMs: number
  },
): Effect.Effect<CommandResult, HarnessFailure> => Effect.gen(function* () {
    const stdoutPath = join(options.home, `.headless-e2e-${options.label}.stdout.log`)
    const stderrPath = join(options.home, `.headless-e2e-${options.label}.stderr.log`)
    const child = yield* Effect.try({
      try: () => Bun.spawn([bun, ...args], {
        cwd: cliDir,
        env: isolatedEnvironment(options.home, options.icnPath),
        stdout: Bun.file(stdoutPath),
        stderr: Bun.file(stderrPath),
      }),
      catch: (cause) => failure("spawn command", cause),
    })
    const waitForExit = (waitMs: number) => Effect.promise(() => child.exited).pipe(
      Effect.timeoutOption(waitMs),
    )
    const kill = (signal: "SIGTERM" | "SIGKILL") => Effect.try({
      try: () => { child.kill(signal) },
      catch: (cause) => failure(`send ${signal} to command ${options.label}`, cause),
    })

    let timedOut = false
    let exitCode = yield* waitForExit(options.timeoutMs)
    if (Option.isNone(exitCode)) {
      timedOut = true
      yield* kill("SIGTERM")
      exitCode = yield* waitForExit(2_000)
    }
    if (Option.isNone(exitCode)) {
      yield* kill("SIGKILL")
      exitCode = yield* waitForExit(5_000)
    }
    if (Option.isNone(exitCode)) {
      return yield* failure(
        "run command",
        new Error(`command ${options.label} did not exit after SIGKILL`),
      )
    }

    const readOutput = (path: string) => Effect.tryPromise({
      try: () => readFile(path, "utf8"),
      catch: (cause) => failure(`read command output ${path}`, cause),
    }).pipe(Effect.catchAll(() => Effect.succeed("")))
    const [stdout, stderr] = yield* Effect.all([
      readOutput(stdoutPath),
      readOutput(stderrPath),
    ], { concurrency: "unbounded" })
    return { exitCode: Option.getOrThrow(exitCode), stdout, stderr, timedOut }
})

const readDirectory = (path: string): Effect.Effect<readonly string[], HarnessFailure> =>
  Effect.tryPromise({
    try: () => readdir(path),
    catch: (cause) => failure(`read directory ${path}`, cause),
  })

const fileExists = (path: string): Effect.Effect<boolean> =>
  Effect.promise(() => Bun.file(path).exists())

const resolveIcnInstallation = Effect.gen(function* () {
    const explicit = process.env.MAGNITUDE_ICN_PATH?.trim()
    if (explicit) {
      if (yield* fileExists(explicit)) return explicit
      return yield* failure(
        "resolve local ICN installation",
        new Error(`MAGNITUDE_ICN_PATH does not exist: ${explicit}`),
      )
    }

    const development = resolve(cliDir, "..", "inference", "target", "development", "installation.json")
    if (yield* fileExists(development)) return development

    return yield* failure(
      "resolve local ICN installation",
      new Error(
        "no local ICN installation found; set MAGNITUDE_ICN_PATH or run the development ICN build first",
      ),
    )
})

const seedIcnCalibrationCache = (home: string): Effect.Effect<void, HarnessFailure> =>
  Effect.gen(function* () {
    const source = join(homedir(), ".magnitude", "cache", "indexes", "hardware-calibrations")
    if (!(yield* fileExists(source))) return
    yield* Effect.tryPromise({
      try: () => cp(
        source,
        join(home, ".magnitude", "cache", "indexes", "hardware-calibrations"),
        { recursive: true },
      ),
      catch: (cause) => failure("seed isolated ICN calibration cache", cause),
    })
  })

const waitForDurableSessionMetadata = (input: {
  readonly path: string
  readonly sessionId: HeadlessSessionId
  readonly workingDirectory: string
  readonly prompt: string
  readonly timeoutMs: number
}): Effect.Effect<boolean> => Effect.gen(function* () {
  const startedAt = yield* Effect.clockWith((clock) => clock.currentTimeMillis)
  const deadline = startedAt + input.timeoutMs
  while (true) {
    const valid = yield* Effect.tryPromise({
      try: () => readFile(input.path, "utf8"),
      catch: (cause) => failure("read durable session metadata", cause),
    }).pipe(
      Effect.flatMap(Schema.decode(Schema.parseJson(Schema.Unknown))),
      Effect.map((metadata) => validateDurableSessionMetadata(metadata, input)),
      Effect.catchAll(() => Effect.succeed(false)),
    )
    if (valid) return true
    const now = yield* Effect.clockWith((clock) => clock.currentTimeMillis)
    if (now >= deadline) return false
    yield* Effect.sleep(25)
  }
})

const program = Effect.gen(function*() {
  const icnPath = yield* resolveIcnInstallation
  let preserveHome = false
  const home = yield* Effect.acquireRelease(
    Effect.tryPromise({
      try: () => mkdtemp(join(tmpdir(), "magnitude-headless-e2e-")),
      catch: (cause) => failure("create temporary home", cause),
    }),
    (path) => preserveHome
      ? Effect.sync(() => process.stderr.write(`headless E2E retained failure HOME: ${path}\n`))
      : Effect.tryPromise({
          try: () => rm(path, { recursive: true, force: true }),
          catch: (cause) => failure("remove temporary home", cause),
        }).pipe(Effect.orDie),
  )

  return yield* Effect.gen(function*() {
  let inferenceRequests = 0
  const rejectedInferenceRequests: string[] = []
  const server = yield* Effect.acquireRelease(
    Effect.try({
      try: () => Bun.serve({
        hostname: "127.0.0.1",
        port: 0,
        async fetch(request) {
          let body: unknown
          try {
            body = await request.json()
          } catch {
            rejectedInferenceRequests.push(`${request.method} ${new URL(request.url).pathname}: invalid JSON`)
            return Response.json({ error: "invalid JSON" }, { status: 400 })
          }
          const pathname = new URL(request.url).pathname
          const expectation = inferenceRequests === 0
            ? { kind: "agent" as const, prompt, systemText: systemOverride }
            : inferenceRequests === 1
              ? { kind: "title" as const, prompt }
              : null
          const validationErrors = expectation === null
            ? ["unexpected additional inference request"]
            : validateFakeInferenceRequest({
                method: request.method,
                pathname,
                authorization: request.headers.get("authorization"),
                body,
              }, expectation)
          if (validationErrors.length > 0) {
            rejectedInferenceRequests.push(`${request.method} ${pathname}: ${validationErrors.join(", ")}`)
            return Response.json({ error: validationErrors.join(", ") }, { status: 400 })
          }
          inferenceRequests += 1
          const chunks = [
            {
              id: "chatcmpl-headless-e2e",
              object: "chat.completion.chunk",
              created: 1,
              model: "fake-model",
              choices: [{
                index: 0,
                delta: { role: "assistant", content: "headless-ok" },
                finish_reason: null,
              }],
            },
            {
              id: "chatcmpl-headless-e2e",
              object: "chat.completion.chunk",
              created: 1,
              model: "fake-model",
              choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
              usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
            },
          ]
          return new Response(
            `${chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("")}data: [DONE]\n\n`,
            { headers: { "content-type": "text/event-stream" } },
          )
        },
      }),
      catch: (cause) => failure("start fake inference server", cause),
    }),
    (resource) => Effect.try({
      try: () => resource.stop(true),
      catch: (cause) => failure("stop fake inference server", cause),
    }).pipe(
      Effect.tapError(() => Effect.sync(() => { preserveHome = true })),
      Effect.orDie,
    ),
  )

  const magnitudeDir = join(home, ".magnitude")
  const stateDir = join(magnitudeDir, "state")
  yield* Effect.tryPromise({
    try: () => mkdir(stateDir, { recursive: true }),
    catch: (cause) => failure("create isolated configuration directory", cause),
  })
  yield* Effect.tryPromise({
    try: () => writeFile(join(magnitudeDir, "config.json"), JSON.stringify({
      providers: {
        fake: {
          displayName: "Fake OpenAI",
          connection: {
            baseUrl: `http://127.0.0.1:${server.port}/v1`,
            authentication: { type: "none" },
          },
          models: {
            "fake-model": {
              displayName: "Fake Model",
              contextWindow: 128_000,
              maxOutputTokens: 8_192,
            },
          },
        },
      },
    })),
    catch: (cause) => failure("write isolated configuration", cause),
  })
  yield* Effect.tryPromise({
    try: () => writeFile(join(stateDir, "onboarding.json"), JSON.stringify({ completed: true })),
    catch: (cause) => failure("write isolated onboarding state", cause),
  })
  yield* Effect.tryPromise({
    try: () => writeFile(join(stateDir, "models.json"), JSON.stringify({
      configurations: [],
      slots: {
        primary: {
          providerId: "custom:fake",
          providerModelId: "fake-model",
          reasoningEffort: "none",
        },
      },
      recentModels: {
        primary: [],
        secondary: [],
      },
      favorites: [],
      configurationRecoveryCompleted: true,
    })),
    catch: (cause) => failure("write isolated model state", cause),
  })
  yield* seedIcnCalibrationCache(home)

  let observedAttestation: AcnProcessAttestation | null = null
  const executeAndVerify = Effect.gen(function*() {
    const command = yield* runCommand([
      "src/index.tsx",
      "--headless",
      "--solo",
      "--system-override",
      systemOverride,
      "--prompt",
      prompt,
    ], { home, icnPath, label: "run", timeoutMs: 120_000 }).pipe(Effect.forkScoped)
    observedAttestation = yield* waitForAcnProcessAttestation(
      join(home, ".magnitude"),
      30_000,
    ).pipe(Effect.mapError((cause) => failure("capture exact ACN owner", cause)))
    const result = yield* Fiber.join(command)

    const sessionsDir = join(magnitudeDir, "sessions")
    const sessionIds = yield* readDirectory(sessionsDir).pipe(Effect.catchAll(() => Effect.succeed([])))
    const persisted = yield* Effect.forEach(sessionIds, (sessionId) =>
      readDirectory(join(sessionsDir, sessionId)).pipe(
        Effect.map((files) => files.includes("meta.json") ? sessionId : null),
        Effect.catchAll(() => Effect.succeed(null)),
      )
    )
    const persistedSessionIds = persisted.filter((sessionId): sessionId is string => sessionId !== null)
    const persistedSessionId = persistedSessionIds[0]
    const headlessSessionId = persistedSessionId === undefined
      ? undefined
      : HeadlessSessionIdSchema.make(persistedSessionId)
    const durableMetadataValid = headlessSessionId !== undefined && (yield* waitForDurableSessionMetadata({
      path: join(sessionsDir, persistedSessionId, "meta.json"),
      sessionId: headlessSessionId,
      workingDirectory: cliDir,
      prompt,
      timeoutMs: 5_000,
    }))
    const stdoutLines = result.stdout.endsWith("\n")
      ? result.stdout.slice(0, -1).split("\n")
      : []
    const deterministicStdout = stdoutLines.length === 3
      && stdoutLines[0] === `> ${prompt}`
      && stdoutLines[1] === "headless-ok"
      && /^✓ Finished · (?:\d+s|\d+m(?: \d+s)?) · 0 tools$/.test(stdoutLines[2] ?? "")
    const deterministicStderr = persistedSessionIds.length === 1
      && result.stderr === `Session: ${persistedSessionIds[0]}\n`

    const failures = [
      result.timedOut ? "CLI timed out" : null,
      result.exitCode !== 0 ? `CLI exited ${result.exitCode}` : null,
      !deterministicStdout ? "stdout did not match the deterministic prompt/answer/summary contract" : null,
      !deterministicStderr ? "stderr did not contain only the durable session receipt" : null,
      ...validateFinalFakeInferenceState(inferenceRequests, rejectedInferenceRequests),
      persistedSessionIds.length !== 1 ? `expected one persisted session, found ${persistedSessionIds.length}` : null,
      !durableMetadataValid ? "durable session metadata did not match the session receipt and CLI cwd" : null,
    ].filter((item): item is string => item !== null)

    if (failures.length > 0) {
      return yield* new HarnessFailure({
        operation: "verify real daemon-backed headless execution",
        message: [
          ...failures,
          `stdout:\n${result.stdout}`,
          `stderr:\n${result.stderr}`,
        ].join("\n"),
      })
    }

    return {
      inferenceRequests,
      persistedSessionId: headlessSessionId!,
    }
  }).pipe(Effect.tapError(() => Effect.sync(() => { preserveHome = true })))

  const verified = yield* Effect.acquireUseRelease(
    Effect.void,
    () => executeAndVerify,
    () => Effect.gen(function*() {
      const stopResult = yield* runCommand(
        ["src/index.tsx", "stop"],
        { home, icnPath, label: "stop", timeoutMs: 30_000 },
      )
      const attestedExit = observedAttestation === null
        ? null
        : yield* attestAcnProcessTreeExit(
            join(home, ".magnitude"),
            observedAttestation,
            5_000,
          ).pipe(Effect.either)
      const failures = [
        stopResult.timedOut ? "stop command timed out" : null,
        stopResult.exitCode !== 0 ? `stop command exited ${stopResult.exitCode}` : null,
        observedAttestation === null ? "could not identify the exact daemon owner while it was live" : null,
        attestedExit?._tag === "Left" ? `could not prove exact daemon tree absence: ${attestedExit.left.message}` : null,
        ...validateFinalFakeInferenceState(inferenceRequests, rejectedInferenceRequests),
      ].filter((item): item is string => item !== null)
      if (failures.length === 0) return

      preserveHome = true
      return yield* new HarnessFailure({
        operation: "stop and verify real daemon",
        message: [
          ...failures,
          `isolated HOME retained at ${home}`,
          `stdout:\n${stopResult.stdout}`,
          `stderr:\n${stopResult.stderr}`,
        ].join("\n"),
      })
    }).pipe(Effect.orDie),
  )
  return verified
  }).pipe(Effect.tapErrorCause(() => Effect.sync(() => { preserveHome = true })))
})

const verified = await Effect.runPromise(Effect.scoped(program))
process.stdout.write(
  `headless E2E passed: exit=0, inferenceRequests=${verified.inferenceRequests}, persistedSession=${verified.persistedSessionId}\n`,
)
