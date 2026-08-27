import { chmod, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"
import { Effect } from "effect"
import { afterEach, describe, expect, it } from "vitest"
import {
  InteractiveProcessFailed,
  interactiveProcessExitCode,
  runInteractiveProcess,
} from "./interactive-process"

const roots: string[] = []

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })))
})

describe("runInteractiveProcess", () => {
  it("passes argv, environment, cwd, and the exit code without a shell", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-interactive-process-"))
    roots.push(root)
    const output = join(root, "result.json")
    const argument = "literal $HOME `echo nope`"

    const termination = await Effect.runPromise(runInteractiveProcess({
      executable: process.execPath,
      args: ["-e", [
        "import { writeFileSync } from 'node:fs'",
        "writeFileSync(process.env.TEST_OUTPUT, JSON.stringify({ argument: process.argv[1], cwd: process.cwd(), marker: process.env.TEST_MARKER, omitted: 'TEST_OMITTED' in process.env }))",
        "process.exit(7)",
      ].join(";"), argument],
      environment: {
        ...process.env,
        TEST_OUTPUT: output,
        TEST_MARKER: "present",
        TEST_OMITTED: undefined,
      },
      workingDirectory: root,
    }))

    expect(termination).toEqual({ _tag: "Exited", code: 7 })
    expect(JSON.parse(await readFile(output, "utf8"))).toEqual({
      argument,
      cwd: await realpath(root),
      marker: "present",
      omitted: false,
    })
  })

  it("preserves signal termination", async () => {
    if (process.platform === "win32") return
    const termination = await Effect.runPromise(runInteractiveProcess({
      executable: process.execPath,
      args: ["-e", "process.kill(process.pid, 'SIGTERM')"],
      environment: process.env,
    }))

    expect(termination).toEqual({ _tag: "Signaled", signal: "SIGTERM" })
    expect(interactiveProcessExitCode(termination)).toBe(143)
  })

  it("reports spawn failures as typed failures", async () => {
    const exit = await Effect.runPromiseExit(runInteractiveProcess({
      executable: join(tmpdir(), "missing-magnitude-interactive-process"),
      args: [],
      environment: process.env,
    }))

    expect(exit._tag).toBe("Failure")
    if (exit._tag === "Failure") {
      const failure = exit.cause.pipe(
        (cause) => cause._tag === "Fail" ? cause.error : undefined,
      )
      expect(failure).toBeInstanceOf(InteractiveProcessFailed)
      expect(failure?.operation).toBe("spawn")
    }
  })

  it("keeps the child in the terminal foreground process group for resize delivery", async () => {
    if (process.platform === "win32") return
    const python = Bun.which("python3")
    if (python === null) return

    const root = await mkdtemp(join(tmpdir(), "magnitude-interactive-pty-"))
    roots.push(root)
    const script = join(root, "pty-check.py")
    const fixtureRoot = join(dirname(fileURLToPath(import.meta.url)), "test-fixtures")
    const runner = join(fixtureRoot, "resize-runner.ts")
    const child = join(fixtureRoot, "resize-child.py")
    await writeFile(script, `
import errno, fcntl, os, pty, select, signal, struct, sys, termios, time

runner, child = sys.argv[1], sys.argv[2]
pid, fd = pty.fork()
if pid == 0:
    os.execv(${JSON.stringify(process.execPath)}, [${JSON.stringify(process.execPath)}, runner, ${JSON.stringify(python)}, child])

def resize(columns, rows):
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))

buffer = b""
def read_until(needle, timeout):
    global buffer
    deadline = time.monotonic() + timeout
    while needle not in buffer and time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], min(0.1, deadline - time.monotonic()))
        if not ready:
            continue
        try:
            chunk = os.read(fd, 65536)
        except OSError as error:
            if error.errno == errno.EIO:
                break
            raise
        if not chunk:
            break
        buffer += chunk
    if needle not in buffer:
        raise RuntimeError("missing " + repr(needle) + " in " + repr(buffer[-1000:]))

def drain(timeout):
    global buffer
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], min(0.05, deadline - time.monotonic()))
        if not ready:
            continue
        try:
            chunk = os.read(fd, 65536)
        except OSError as error:
            if error.errno == errno.EIO:
                return
            raise
        if not chunk:
            return
        buffer += chunk

try:
    resize(120, 40)
    read_until(b"READY", 10)
    drain(0.5)
    resize(121, 41)
    read_until(b"SIZE 121 41", 5)
    resize(70, 30)
    read_until(b"SIZE 70 30", 5)
    resize(132, 44)
    read_until(b"SIZE 132 44", 5)
    print("resize-delivered")
finally:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass
    os.close(fd)
`)
    await chmod(script, 0o700)

    const result = Bun.spawnSync({
      cmd: [python, script, runner, child],
      stdout: "pipe",
      stderr: "pipe",
    })
    const stdout = new TextDecoder().decode(result.stdout)
    const stderr = new TextDecoder().decode(result.stderr)
    expect(result.exitCode, `${stderr}\n${stdout}`).toBe(0)
    expect(stdout).toContain("resize-delivered")
  }, 25_000)
})
