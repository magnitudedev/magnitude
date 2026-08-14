import { EventEmitter } from "node:events"
import { describe, expect, it, vi } from "vitest"
import { Effect, Layer, Option, Stream } from "effect"
import type {
  DisplayMessage,
  DisplayRootStatus,
  DisplayTimelineEntry,
  DisplayViewShape,
  DisplayViewStateEvent,
  SessionMetadata,
  SessionOptions,
  StreamEvent,
} from "@magnitudedev/sdk"
import { makeDisplayViewSnapshotFixture } from "@magnitudedev/sdk/testing"
import {
  HeadlessSessionClient,
  HeadlessSessionIdSchema,
  type Platform,
} from "@magnitudedev/client-common"
import { runHeadless as runHeadlessEffect, type RunHeadlessOptions } from "./headless"

const runHeadless = (
  options: Parameters<typeof runHeadlessEffect>[0],
  dependencies: Parameters<typeof runHeadlessEffect>[1],
) => Effect.runPromise(runHeadlessEffect(options, {
  ...dependencies,
  makeSessionId: dependencies.makeSessionId
    ?? (() => HeadlessSessionIdSchema.make(metadata.sessionId)),
}))

const metadata: SessionMetadata = {
  sessionId: "session-1",
  title: "Headless test",
  cwd: "/repo",
  createdAt: 1,
  updatedAt: 2,
  messageCount: 2,
  lastMessage: "done",
}

const userMessage: DisplayMessage = {
  id: "user-1",
  type: "user_message",
  content: "test prompt",
  timestamp: 1,
  taskMode: false,
  attachments: [],
}

const assistantMessage: DisplayMessage = {
  id: "assistant-1",
  type: "assistant_message",
  content: "done",
  timestamp: 3,
}

const shellPresentation: Extract<DisplayTimelineEntry, { readonly kind: "tool_step" }>["step"] = {
  toolKey: "shell",
  phase: "completed",
  tone: "success",
  icon: "terminal",
  command: "printf done",
  done: "completed",
  exitCode: 0,
  pid: null,
  stdout: "done",
  stderr: "",
  partialStdout: "",
  partialStderr: "",
  stdoutPath: null,
  stderrPath: null,
  errorText: null,
  running: false,
  failed: false,
}

const toolMessage: DisplayMessage = {
  id: "tool-message-1",
  type: "tool",
  toolKey: "shell",
  cluster: Option.none(),
  presentation: Option.some(shellPresentation),
  filter: Option.none(),
  resultFilePath: Option.none(),
  timestamp: 2,
}

function entry(message: DisplayMessage): DisplayTimelineEntry {
  return {
    kind: "message",
    id: `entry:${message.id}`,
    messageId: message.id,
    timestamp: message.timestamp,
    role: message.type === "user_message" ? "user" : "assistant",
    streaming: false,
    interrupted: false,
    nextMessageInterrupted: false,
  }
}

function stateEvent(
  shape: DisplayViewShape,
  status: DisplayRootStatus,
  messages: readonly DisplayMessage[],
  entries: readonly DisplayTimelineEntry[],
): DisplayViewStateEvent {
  return {
    _tag: "state",
    ...makeDisplayViewSnapshotFixture({
      shape,
      session: { sessionId: metadata.sessionId, title: metadata.title ?? "", cwd: metadata.cwd },
      status,
      messages,
      entries,
      mode: status._tag === "Working" ? "streaming" : "idle",
      streamingMessageId: null,
    }),
  }
}

function successfulClient(
  createdOptions: SessionOptions[],
  createdInitial: Array<{ readonly type: "message" | "goal"; readonly content: string }> = [],
  terminalMessage: DisplayMessage = assistantMessage,
  createdSessionIds: string[] = [],
): HeadlessSessionClient {
  let shape: DisplayViewShape | null = null
  const toolEntry: DisplayTimelineEntry = {
    kind: "tool_step",
    id: "tool-entry-1",
    messageId: "tool-message-1",
    timestamp: 2,
    step: shellPresentation,
  }

  return {
    createSession: (request) => Effect.sync(() => {
      createdOptions.push(request.options)
      createdInitial.push(request.initial)
      createdSessionIds.push(request.sessionId)
      return { _tag: "created" as const, metadata }
    }),
    getSession: () => Effect.succeed(metadata),
    resyncDisplayView: () => Effect.sync(() => {
      if (!shape) throw new Error("shape must be set before resync")
      return stateEvent(
        shape,
        { _tag: "Worked", lastProductiveMs: 20 },
        [userMessage, toolMessage, terminalMessage],
        [entry(userMessage), toolEntry, entry(terminalMessage)],
      )
    }),
    streamDisplayView: (_sessionId, _viewId, requestedShape) => {
      shape = requestedShape
      return Stream.make(stateEvent(
        requestedShape,
        { _tag: "Worked", lastProductiveMs: 20 },
        [userMessage, toolMessage, terminalMessage],
        [entry(userMessage), toolEntry, entry(terminalMessage)],
      ))
    },
  }
}

function writer() {
  let value = ""
  return {
    stream: {
      write: (chunk: string, callback?: (error?: Error | null) => void) => {
        value += chunk
        callback?.()
        return true
      },
    },
    read: () => value,
  }
}

function readyPlatform(onShutdown: () => void): Platform {
  const ready = { _tag: "Ready" as const }
  return {
    acnStartup: {
      prepare: Effect.succeed(ready),
      retry: Effect.void,
      state: { get: Effect.succeed(ready), changes: Stream.empty },
    },
    protocolLayer: Layer.empty,
    shutdown: async () => {
      onShutdown()
    },
  } as unknown as Platform
}

function failedPlatform(onShutdown: () => void, message = "daemon unavailable"): Platform {
  const failed = {
    _tag: "Failed" as const,
    stage: "LaunchDaemon" as const,
    message,
    retryable: true,
  }
  return {
    acnStartup: {
      prepare: Effect.succeed(failed),
      retry: Effect.void,
      state: { get: Effect.succeed(failed), changes: Stream.empty },
    },
    protocolLayer: Layer.empty,
    shutdown: async () => {
      onShutdown()
    },
  } as unknown as Platform
}

const baseOptions: RunHeadlessOptions = {
  sessionStart: { _tag: "new" },
  initialPrompt: "test prompt",
  debug: false,
  solo: true,
  autopilot: false,
  disableCwdSafeguards: true,
  disableShellSafeguards: true,
  atifPath: "/tmp/run.atif",
  systemOverride: "system override",
}

describe("runHeadless", () => {
  it("runs the daemon-backed client path, renders stdout, persists the session, and cleans up", async () => {
    const stdout = writer()
    const stderr = writer()
    const createdOptions: SessionOptions[] = []
    const createdSessionIds: string[] = []
    let shutdown = false
    const invoke = runHeadless

    const exitCode = await invoke(baseOptions, {
      createPlatform: async () => readyPlatform(() => { shutdown = true }),
      sessionClientLayer: Layer.succeed(
        HeadlessSessionClient,
        successfulClient(createdOptions, [], assistantMessage, createdSessionIds),
      ),
      stdout: stdout.stream,
      stderr: stderr.stream,
      cwd: () => "/repo",
      registerSignalHandlers: false,
      makeSessionId: () => HeadlessSessionIdSchema.make(metadata.sessionId),
    })

    expect(exitCode).toBe(0)
    expect(shutdown).toBe(true)
    expect(stdout.read()).toContain("> test prompt")
    expect(stdout.read()).toContain("$ printf done · exit 0")
    expect(stdout.read()).toContain("done")
    expect(stdout.read()).toContain("Finished")
    expect(stderr.read()).toContain(metadata.sessionId)
    expect(createdSessionIds).toEqual([metadata.sessionId])
    expect(createdOptions).toEqual([{
      headless: true,
      solo: true,
      disableCwdSafeguards: true,
      disableShellSafeguards: true,
      atifPath: "/tmp/run.atif",
      systemPromptOverride: "system override",
    }])
  })

  it("owns termination signals while the platform is being acquired", async () => {
    const listeners = new Map<"SIGINT" | "SIGTERM", () => void>()
    const signalTarget = {
      once: (signal: "SIGINT" | "SIGTERM", listener: () => void) => {
        listeners.set(signal, listener)
      },
      removeListener: (signal: "SIGINT" | "SIGTERM", listener: () => void) => {
        if (listeners.get(signal) === listener) listeners.delete(signal)
      },
    }
    const createdOptions: SessionOptions[] = []
    let shutdown = false

    const exitCode = await runHeadless(baseOptions, {
      createPlatform: async () => {
        const terminate = listeners.get("SIGINT")
        if (!terminate) throw new Error("SIGINT was not owned before platform acquisition")
        terminate()
        return readyPlatform(() => { shutdown = true })
      },
      sessionClientLayer: Layer.succeed(HeadlessSessionClient, successfulClient(createdOptions)),
      stdout: writer().stream,
      stderr: writer().stream,
      cwd: () => "/repo",
      signalTarget,
    })

    expect(exitCode).toBe(130)
    expect(createdOptions).toEqual([])
    expect(shutdown).toBe(true)
    expect(listeners.size).toBe(0)
  })

  it("emits a client-known session receipt when CreateSession never replies after dispatch", async () => {
    const listeners = new Map<NodeJS.Signals, () => void>()
    const stderr = writer()
    const assigned = HeadlessSessionIdSchema.make("session-ambiguous")
    const dispatched: string[] = []
    const signalTarget = {
      once: (signal: NodeJS.Signals, listener: () => void) => {
        listeners.set(signal, listener)
      },
      removeListener: (signal: NodeJS.Signals, listener: () => void) => {
        if (listeners.get(signal) === listener) listeners.delete(signal)
      },
    }
    const client: HeadlessSessionClient = {
      ...successfulClient([]),
      createSession: (request) => Effect.sync(() => {
        dispatched.push(request.sessionId)
        queueMicrotask(() => listeners.get("SIGINT")?.())
      }).pipe(Effect.zipRight(Effect.never)),
    }

    const exitCode = await runHeadless(baseOptions, {
      createPlatform: async () => readyPlatform(() => {}),
      sessionClientLayer: Layer.succeed(HeadlessSessionClient, client),
      stdout: writer().stream,
      stderr: stderr.stream,
      cwd: () => "/repo",
      signalTarget,
      makeSessionId: () => assigned,
      fiberInterruptTimeoutMs: 10,
    })

    expect(exitCode).toBe(130)
    expect(dispatched).toEqual([assigned])
    expect(stderr.read()).toBe(`Session: ${assigned}\n`)
    expect(listeners.size).toBe(0)
  })

  it("waits for stdout backpressure before completing", async () => {
    let release: (() => void) | null = null
    let settled = false
    const stdout = {
      write: (_chunk: string, callback?: (error?: Error | null) => void) => {
        if (release === null) {
          release = () => callback?.()
          return false
        }
        callback?.()
        return true
      },
    }
    const completion = runHeadless(baseOptions, {
      createPlatform: async () => readyPlatform(() => {}),
      sessionClientLayer: Layer.succeed(HeadlessSessionClient, successfulClient([])),
      stdout,
      stderr: writer().stream,
      cwd: () => "/repo",
      registerSignalHandlers: false,
      makeSessionId: () => HeadlessSessionIdSchema.make(metadata.sessionId),
    }).then((exitCode) => {
      settled = true
      return exitCode
    })

    await Bun.sleep(10)
    expect(settled).toBe(false)
    expect(release).not.toBeNull()
    ;(release as unknown as () => void)()
    expect(await completion).toBe(0)
  })

  it("routes asynchronous EPIPE callbacks and error events through Effect teardown", async () => {
    let shutdown = false
    const events = new EventEmitter()
    const stdout = {
      write: (_chunk: string, callback?: (error?: Error | null) => void) => {
        const error = Object.assign(new Error("broken pipe"), { code: "EPIPE" })
        queueMicrotask(() => {
          callback?.(error)
          events.emit("error", error)
        })
        return false
      },
      once: events.once.bind(events),
      off: events.off.bind(events),
    }

    const exitCode = await runHeadless(baseOptions, {
      createPlatform: async () => readyPlatform(() => { shutdown = true }),
      sessionClientLayer: Layer.succeed(HeadlessSessionClient, successfulClient([])),
      stdout,
      stderr: writer().stream,
      cwd: () => "/repo",
      registerSignalHandlers: false,
      makeSessionId: () => HeadlessSessionIdSchema.make(metadata.sessionId),
    })

    expect(exitCode).toBe(1)
    expect(shutdown).toBe(true)
  })

  it("does not let a stalled platform acquisition make signal exit unbounded", async () => {
    const listeners = new Map<NodeJS.Signals, () => void>()
    const signalTarget = {
      once: (signal: NodeJS.Signals, listener: () => void) => {
        listeners.set(signal, listener)
        if (signal === "SIGINT") queueMicrotask(listener)
      },
      removeListener: (signal: NodeJS.Signals, listener: () => void) => {
        if (listeners.get(signal) === listener) listeners.delete(signal)
      },
    }

    const exitCode = await Promise.race([
      runHeadless(baseOptions, {
        createPlatform: () => new Promise<Platform>(() => undefined),
        sessionClientLayer: Layer.succeed(HeadlessSessionClient, successfulClient([])),
        stdout: { write: () => true },
        stderr: { write: () => true },
        registerSignalHandlers: true,
        signalTarget,
      }),
      Bun.sleep(100).then(() => -1),
    ])

    expect(exitCode).toBe(130)
    expect(listeners.size).toBe(0)
  })

  it("does not let uninterruptible daemon preparation make signal exit unbounded", async () => {
    const listeners = new Map<NodeJS.Signals, () => void>()
    const signalTarget = {
      once: (signal: NodeJS.Signals, listener: () => void) => {
        listeners.set(signal, listener)
      },
      removeListener: (signal: NodeJS.Signals, listener: () => void) => {
        if (listeners.get(signal) === listener) listeners.delete(signal)
      },
    }
    const base = readyPlatform(() => {})
    const platform = {
      ...base,
      acnStartup: {
        ...base.acnStartup,
        prepare: Effect.uninterruptible(
          Effect.sync(() => listeners.get("SIGTERM")?.()).pipe(
            Effect.zipRight(Effect.never),
          ),
        ),
      },
    } as Platform

    const exitCode = await Promise.race([
      runHeadless(baseOptions, {
        createPlatform: async () => platform,
        sessionClientLayer: Layer.succeed(HeadlessSessionClient, successfulClient([])),
        stdout: writer().stream,
        stderr: writer().stream,
        signalTarget,
        fiberInterruptTimeoutMs: 10,
        platformShutdownTimeoutMs: 10,
      }),
      Bun.sleep(100).then(() => -1),
    ])

    expect(exitCode).toBe(143)
    expect(listeners.size).toBe(0)
  })

  it("waits boundedly for a late platform and shuts it down before returning", async () => {
    const listeners = new Map<NodeJS.Signals, () => void>()
    const signalTarget = {
      once: (signal: NodeJS.Signals, listener: () => void) => {
        listeners.set(signal, listener)
        if (signal === "SIGTERM") queueMicrotask(listener)
      },
      removeListener: (signal: NodeJS.Signals, listener: () => void) => {
        if (listeners.get(signal) === listener) listeners.delete(signal)
      },
    }
    let shutdown = false

    const exitCode = await runHeadless(baseOptions, {
      createPlatform: () => Bun.sleep(5).then(() => readyPlatform(() => { shutdown = true })),
      sessionClientLayer: Layer.succeed(HeadlessSessionClient, successfulClient([])),
      stdout: writer().stream,
      stderr: writer().stream,
      signalTarget,
      latePlatformAcquisitionTimeoutMs: 50,
      platformShutdownTimeoutMs: 20,
    })

    expect(exitCode).toBe(143)
    expect(shutdown).toBe(true)
    expect(listeners.size).toBe(0)
  })

  it("bounds acquired platform shutdown", async () => {
    const stderr = writer()
    const platform = {
      ...readyPlatform(() => {}),
      shutdown: () => new Promise<never>(() => undefined),
    } as Platform

    const exitCode = await Promise.race([
      runHeadless(baseOptions, {
        createPlatform: async () => platform,
        sessionClientLayer: Layer.succeed(HeadlessSessionClient, successfulClient([])),
        stdout: writer().stream,
        stderr: stderr.stream,
        registerSignalHandlers: false,
        platformShutdownTimeoutMs: 10,
      }),
      Bun.sleep(100).then(() => -1),
    ])

    expect(exitCode).toBe(1)
    expect(stderr.read()).toContain("timed out while releasing the daemon client")
  })

  it("preserves signal status when acquired platform shutdown stalls", async () => {
    const listeners = new Map<NodeJS.Signals, () => void>()
    const signalTarget = {
      once: (signal: NodeJS.Signals, listener: () => void) => {
        listeners.set(signal, listener)
      },
      removeListener: (signal: NodeJS.Signals, listener: () => void) => {
        if (listeners.get(signal) === listener) listeners.delete(signal)
      },
    }
    const platform = {
      ...readyPlatform(() => {}),
      shutdown: () => {
        listeners.get("SIGINT")?.()
        return new Promise<never>(() => undefined)
      },
    } as Platform

    const exitCode = await Promise.race([
      runHeadless(baseOptions, {
        createPlatform: async () => platform,
        sessionClientLayer: Layer.succeed(HeadlessSessionClient, successfulClient([])),
        stdout: writer().stream,
        stderr: writer().stream,
        signalTarget,
        platformShutdownTimeoutMs: 10,
      }),
      Bun.sleep(100).then(() => -1),
    ])

    expect(exitCode).toBe(130)
    expect(listeners.size).toBe(0)
  })

  it("bounds post-signal output and arms exit scheduling immediately", async () => {
    const listeners = new Map<NodeJS.Signals, () => void>()
    const scheduled: number[] = []
    const signalTarget = {
      once: (signal: NodeJS.Signals, listener: () => void) => listeners.set(signal, listener),
      removeListener: (signal: NodeJS.Signals, listener: () => void) => {
        if (listeners.get(signal) === listener) listeners.delete(signal)
      },
    }
    const client: HeadlessSessionClient = {
      ...successfulClient([]),
      createSession: () => Effect.sync(() => queueMicrotask(() => listeners.get("SIGINT")?.())).pipe(
        Effect.zipRight(Effect.never),
      ),
    }
    const blocked = { write: (_chunk: string, _callback: (error?: Error | null) => void) => false }
    const exitCode = await Promise.race([
      runHeadless(baseOptions, {
        createPlatform: async () => readyPlatform(() => {}),
        sessionClientLayer: Layer.succeed(HeadlessSessionClient, client),
        stdout: blocked,
        stderr: writer().stream,
        signalTarget,
        signalOutputTimeoutMs: 10,
        onTerminationSignal: (code) => scheduled.push(code),
      }),
      Bun.sleep(100).then(() => -1),
    ])
    expect(exitCode).toBe(130)
    expect(scheduled).toEqual([130])
    expect(listeners.size).toBe(0)
  })

  it("cleans a platform that resolves after the late-acquisition grace window", async () => {
    const listeners = new Map<NodeJS.Signals, () => void>()
    const signalTarget = {
      once: (signal: NodeJS.Signals, listener: () => void) => {
        listeners.set(signal, listener)
        if (signal === "SIGTERM") queueMicrotask(listener)
      },
      removeListener: (signal: NodeJS.Signals, listener: () => void) => {
        if (listeners.get(signal) === listener) listeners.delete(signal)
      },
    }
    let shutdown = false
    const exitCode = await runHeadless(baseOptions, {
      createPlatform: () => Bun.sleep(40).then(() => readyPlatform(() => { shutdown = true })),
      sessionClientLayer: Layer.succeed(HeadlessSessionClient, successfulClient([])),
      stdout: writer().stream,
      stderr: writer().stream,
      signalTarget,
      latePlatformAcquisitionTimeoutMs: 5,
      platformShutdownTimeoutMs: 20,
    })
    expect(exitCode).toBe(143)
    expect(shutdown).toBe(false)
    await Bun.sleep(80)
    expect(shutdown).toBe(true)
  })

  it("starts goal work through the same durable session creation path", async () => {
    const createdOptions: SessionOptions[] = []
    const createdInitial: Array<{ readonly type: "message" | "goal"; readonly content: string }> = []
    const invoke = runHeadless
    const { initialPrompt: _initialPrompt, ...optionsWithoutPrompt } = baseOptions

    const exitCode = await invoke({
      ...optionsWithoutPrompt,
      goal: "ship the fix",
    }, {
      createPlatform: async () => readyPlatform(() => {}),
      sessionClientLayer: Layer.succeed(
        HeadlessSessionClient,
        successfulClient(createdOptions, createdInitial),
      ),
      stdout: writer().stream,
      stderr: writer().stream,
      cwd: () => "/repo",
      registerSignalHandlers: false,
    })

    expect(exitCode).toBe(0)
    expect(createdInitial).toEqual([{ type: "goal", content: "ship the fix" }])
  })

  it("keeps successful display resync diagnostics out of process output", async () => {
    const stdout = writer()
    const stderr = writer()
    const invalidPatch: StreamEvent = {
      _tag: "patch",
      ops: [{
        op: "replace",
        path: ["state", "session", "sessionId", "nested"],
        value: "invalid",
      }],
    }
    const baseClient = successfulClient([])
    const client: HeadlessSessionClient = {
      ...baseClient,
      streamDisplayView: (sessionId, viewId, shape) => baseClient
        .streamDisplayView(sessionId, viewId, shape)
        .pipe(Stream.take(1), Stream.concat(Stream.make(invalidPatch))),
    }
    const consoleCalls: unknown[][] = []
    const captureConsoleCall = (...args: unknown[]) => { consoleCalls.push(args) }
    const consoleSpies = [
      vi.spyOn(console, "log").mockImplementation(captureConsoleCall),
      vi.spyOn(console, "warn").mockImplementation(captureConsoleCall),
      vi.spyOn(console, "error").mockImplementation(captureConsoleCall),
    ]

    try {
      const exitCode = await runHeadless(baseOptions, {
        createPlatform: async () => readyPlatform(() => {}),
        sessionClientLayer: Layer.succeed(HeadlessSessionClient, client),
        stdout: stdout.stream,
        stderr: stderr.stream,
        cwd: () => "/repo",
        registerSignalHandlers: false,
      })

      expect(exitCode).toBe(0)
      expect(stdout.read()).toContain("Finished")
      expect(stderr.read()).toBe(`Session: ${metadata.sessionId}\n`)
      expect(consoleCalls).toEqual([])
    } finally {
      for (const spy of consoleSpies) spy.mockRestore()
    }
  })

  it("returns a failing process status for an authoritative terminal error", async () => {
    const stderr = writer()
    const stdout = writer()
    const failure: DisplayMessage = {
      id: "error-1",
      type: "error",
      message: "provider not ready",
      timestamp: 3,
      cta: Option.none(),
    }
    const invoke = runHeadless

    const exitCode = await invoke(baseOptions, {
      createPlatform: async () => readyPlatform(() => {}),
      sessionClientLayer: Layer.succeed(
        HeadlessSessionClient,
        successfulClient([], [], failure),
      ),
      stdout: stdout.stream,
      stderr: stderr.stream,
      cwd: () => "/repo",
      registerSignalHandlers: false,
    })

    expect(exitCode).toBe(1)
    expect(stdout.read()).toContain("provider not ready")
    expect(stdout.read()).toContain("Failed")
  })

  it("reports daemon startup failure and still releases platform resources", async () => {
    const stderr = writer()
    let shutdown = false
    const invoke = runHeadless

    const exitCode = await invoke(baseOptions, {
      createPlatform: async () => failedPlatform(() => { shutdown = true }),
      sessionClientLayer: Layer.succeed(HeadlessSessionClient, successfulClient([])),
      stdout: writer().stream,
      stderr: stderr.stream,
      cwd: () => "/repo",
      registerSignalHandlers: false,
    })

    expect(exitCode).toBe(1)
    expect(shutdown).toBe(true)
    expect(stderr.read()).toContain("daemon unavailable")
  })

  it("neutralizes terminal controls and embedded line breaks in daemon failures", async () => {
    const stderr = writer()

    const exitCode = await runHeadless(baseOptions, {
      createPlatform: async () => failedPlatform(
        () => {},
        "safe\u001b]52;c;Y2xpcGJvYXJk\u0007\u001b[31mred\u001b[0m\rrewrite\b\tend\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u2028split\u2029paragraph\nSession: forged",
      ),
      sessionClientLayer: Layer.succeed(HeadlessSessionClient, successfulClient([])),
      stdout: writer().stream,
      stderr: stderr.stream,
      cwd: () => "/repo",
      registerSignalHandlers: false,
    })

    expect(exitCode).toBe(1)
    expect(stderr.read()).toBe("Error: ACN LaunchDaemon failed: safered\\nrewrite end\\u061c\\u200e\\u200f\\u202a\\u202b\\u202c\\u202d\\u202e\\u2066\\u2067\\u2068\\u2069\\u2028split\\u2029paragraph\\nSession: forged\n")
  })

  it("rejects unsupported or ambiguous options before starting the daemon", async () => {
    const stderr = writer()
    let platformStarts = 0
    const invoke = runHeadless
    const dependencies = {
      createPlatform: async () => {
        platformStarts += 1
        return readyPlatform(() => {})
      },
      stdout: writer().stream,
      stderr: stderr.stream,
      cwd: () => "/repo",
      registerSignalHandlers: false,
    }

    expect(await invoke({ ...baseOptions, sessionStart: { _tag: "latest" } }, dependencies)).toBe(2)
    expect(await invoke({
      ...baseOptions,
      sessionStart: { _tag: "resume", sessionId: "session-old" },
    }, dependencies)).toBe(2)
    expect(await invoke({ ...baseOptions, autopilot: true }, dependencies)).toBe(2)
    expect(await invoke({ ...baseOptions, goal: "goal" }, dependencies)).toBe(2)
    expect(await invoke({ ...baseOptions, initialPrompt: undefined }, dependencies)).toBe(2)
    expect(await invoke({ ...baseOptions, setup: true }, dependencies)).toBe(2)
    expect(platformStarts).toBe(0)
    expect(stderr.read()).toContain("headless")
  })

  it("does not emit invalid-option output until its Effect executes", async () => {
    const stderr = writer()
    let platformStarts = 0
    const effect = runHeadlessEffect(
      { ...baseOptions, sessionStart: { _tag: "latest" } },
      {
        createPlatform: async () => {
          platformStarts += 1
          throw new Error("must not start")
        },
        stderr: stderr.stream,
        registerSignalHandlers: false,
      },
    )

    expect(stderr.read()).toBe("")
    expect(platformStarts).toBe(0)
    expect(await Effect.runPromise(effect)).toBe(2)
    expect(stderr.read()).toContain("--resume is not supported")
    expect(platformStarts).toBe(0)
  })

  it("rejects structurally invalid external options before starting the platform", async () => {
    const stderr = writer()
    let platformStarts = 0
    const malformedOptions = { ...baseOptions, initialPrompt: 42 }
    const effect = runHeadlessEffect(
      // @ts-expect-error External JavaScript callers can violate the TypeScript interface.
      malformedOptions,
      {
        createPlatform: async () => {
          platformStarts += 1
          throw new Error("must not start")
        },
        stderr: stderr.stream,
        registerSignalHandlers: false,
      },
    )

    expect(stderr.read()).toBe("")
    expect(await Effect.runPromise(effect)).toBe(2)
    expect(stderr.read()).toBe("Error: invalid --headless options\n")
    expect(platformStarts).toBe(0)
  })


  it("rejects excess top-level and nested external options", async () => {
    let platformStarts = 0
    const dependencies = {
      createPlatform: async () => {
        platformStarts += 1
        return readyPlatform(() => {})
      },
      sessionClientLayer: Layer.succeed(HeadlessSessionClient, successfulClient([])),
      stdout: writer().stream,
      stderr: writer().stream,
      registerSignalHandlers: false,
    }
    const topLevel = { ...baseOptions, forged: true }
    const nested = { ...baseOptions, sessionStart: { _tag: "new" as const, forged: true } }

    expect(await runHeadlessEffect(
      topLevel,
      dependencies,
    ).pipe(Effect.runPromise)).toBe(2)
    expect(await runHeadlessEffect(
      nested,
      dependencies,
    ).pipe(Effect.runPromise)).toBe(2)
    expect(platformStarts).toBe(0)
  })
})
