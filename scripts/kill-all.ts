import { Data, Effect } from "effect"

export type MagnitudeProcessKind = "ICN" | "ACN" | "CLI"

export interface ProcessInfo {
  readonly pid: number
  readonly parentPid: number
  readonly command: string
}

export interface MagnitudeProcess extends ProcessInfo {
  readonly kind: MagnitudeProcessKind
}

class ProcessListError extends Data.TaggedError("ProcessListError")<{
  readonly cause?: unknown
  readonly message: string
}> {}

const executable = (name: string): RegExp =>
  new RegExp(`^\\s*(?:\\S*[/\\\\])?${name}(?:\\.exe)?(?:\\s|$)`, "i")

const scriptInvocation = (path: string): RegExp =>
  new RegExp(
    `(?:^|\\s)(?:\\S*[/\\\\])?(?:bun|node)(?:\\.exe)?(?:\\s+run)?\\s+\\S*${path.replaceAll("/", "[/\\\\]")}(?:\\s|$)`,
    "i",
  )

const icnExecutable = executable("(?:magnitude-icn|icn-server)")
const acnExecutable = executable("magnitude-acn")
const cliExecutable = executable("magnitude-cli")
const acnSource = scriptInvocation("packages/acn/src/binary\\.ts")
const cliSource = scriptInvocation("cli/src/index\\.tsx")
const npmCliSource = scriptInvocation("packages/cli/bin/magnitude\\.js")

export const classifyMagnitudeProcess = (
  command: string,
): MagnitudeProcessKind | undefined => {
  if (
    icnExecutable.test(command) ||
    /(?:^|\s)(?:icn:dev|icn:serve)(?:\s|$)/i.test(command) ||
    /(?:^|\s)-p\s+icn-server(?:\s|$)/i.test(command)
  ) return "ICN"

  if (acnExecutable.test(command) || (acnSource.test(command) && /(?:^|\s)serve(?:\s|$)/.test(command))) {
    return "ACN"
  }

  if (cliExecutable.test(command) || cliSource.test(command) || npmCliSource.test(command)) {
    return "CLI"
  }

  return undefined
}

export const parseProcessList = (output: string): readonly ProcessInfo[] =>
  output.split("\n").flatMap((line) => {
    const match = /^\s*(\d+)\s+(\d+)\s+(.+)$/.exec(line)
    if (!match) return []
    return [{
      pid: Number(match[1]),
      parentPid: Number(match[2]),
      command: match[3],
    }]
  })

const collectProcesses = Effect.tryPromise({
  try: async () => {
    const processList = Bun.spawn(["ps", "-axo", "pid=,ppid=,command="], {
      stdout: "pipe",
      stderr: "pipe",
    })
    const [exitCode, stdout, stderr] = await Promise.all([
      processList.exited,
      new Response(processList.stdout).text(),
      new Response(processList.stderr).text(),
    ])
    if (exitCode !== 0) {
      throw new ProcessListError({
        message: `ps exited with code ${exitCode}: ${stderr.trim()}`,
      })
    }
    return parseProcessList(stdout)
  },
  catch: (cause) => cause instanceof ProcessListError
    ? cause
    : new ProcessListError({ cause, message: "Could not list processes" }),
})

const isRunning = (pid: number): boolean => {
  try {
    process.kill(pid, 0)
    return true
  } catch (cause) {
    return cause instanceof Error && "code" in cause && cause.code === "EPERM"
  }
}

const signal = (target: MagnitudeProcess, name: NodeJS.Signals): void => {
  try {
    process.kill(target.pid, name)
    console.log(`${name === "SIGTERM" ? "Stopping" : "Killing"} ${target.kind} pid ${target.pid}: ${target.command}`)
  } catch (cause) {
    if (!(cause instanceof Error && "code" in cause && cause.code === "ESRCH")) throw cause
  }
}

const waitForExit = (targets: readonly MagnitudeProcess[], timeoutMs: number) =>
  Effect.gen(function* () {
    const deadline = Date.now() + timeoutMs
    let running = targets.filter(({ pid }) => isRunning(pid))
    while (running.length > 0 && Date.now() < deadline) {
      yield* Effect.sleep("50 millis")
      running = running.filter(({ pid }) => isRunning(pid))
    }
    return running
  })

export const killAllMagnitudeProcesses = Effect.gen(function* () {
  const ownPid = process.pid
  const targets = (yield* collectProcesses).flatMap((candidate): readonly MagnitudeProcess[] => {
    if (candidate.pid === ownPid) return []
    const kind = classifyMagnitudeProcess(candidate.command)
    return kind ? [{ ...candidate, kind }] : []
  })

  if (targets.length === 0) {
    console.log("No Magnitude ICN, ACN, or CLI processes found.")
    return
  }

  for (const target of targets) signal(target, "SIGTERM")
  const remaining = yield* waitForExit(targets, 2_000)
  for (const target of remaining) signal(target, "SIGKILL")
  yield* waitForExit(remaining, 1_000)
})

if (import.meta.main) {
  await Effect.runPromise(killAllMagnitudeProcesses)
}
