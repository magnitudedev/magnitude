import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Deferred, Effect, Fiber, Layer, Schema } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { NativeTerminalDriver, TerminalConfig, TerminalDriver, TerminalScreen, waitForTerminal } from "../src/terminal"

const runtime = process.env.LAB_TERMINAL_NODE_EXECUTABLE
const fixture = `
if (!process.stdin.isTTY || !process.stdout.isTTY) process.exit(72)
process.stdin.setRawMode(true); process.stdin.resume()
process.stdout.write('\\x1b[2J\\x1b[HBEFORE\\r\\nTTY=true\\r\\n')
process.stdout.on('resize', () => process.stdout.write('SIZE:' + process.stdout.columns + 'x' + process.stdout.rows + '\\r\\n'))
let input = ''
process.stdin.on('data', bytes => {
 for (const ch of bytes.toString()) {
  if (ch === '\\x03') { process.stdout.write('INTERRUPTED\\r\\n'); input = ''; continue }
  if (ch === '\\x04') { process.stdout.write('EXITED\\r\\n'); process.exit(0) }
  if (ch === '\\r') {
   if (input === 'size') process.stdout.write('SIZE:' + process.stdout.getWindowSize().join('x') + '\\r\\n')
   if (input === 'paint') process.stdout.write('\\x1b[2J\\x1b[HAFTER \\x1b[31mcolored\\x1b[0m 漢字\\r\\n')
   if (input === 'alternate') process.stdout.write('\\x1b[?1049h\\x1b[HALTERNATE\\r\\n')
   if (input === 'normal') process.stdout.write('\\x1b[?1049l')
   if (input === 'again') process.stdout.write('RECOVERED\\r\\n')
   input = ''; continue
  }
  input += ch
 }
})
`
const config = (root: string, script: string, evidence: string) => TerminalConfig.make({ runtime: runtime!, executable: process.execPath, args: [script], cwd: root,
  environment: { PATH: process.env.PATH ?? "", HOME: root, ...(process.env.SystemRoot ? { SystemRoot: process.env.SystemRoot } : {}) },
  evidence: join(root, evidence), columns: 80, rows: 24 })

test.skipIf(!runtime)("native PTY renders cursor movement, accepts keyboard interruption, resizes and exits normally", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-terminal-" }), script = join(root, "fixture.cjs")
  yield* fs.writeFileString(script, fixture)
  const cleanup: string[] = []
  yield* Effect.scoped(Effect.gen(function* () {
    const terminal = yield* (yield* TerminalDriver).start(config(root, script, "evidence"), message => { cleanup.push(message) })
    yield* waitForTerminal(terminal, s => s.lines.some(line => line.includes("TTY=true")), "establish a real terminal")
    yield* terminal.resize(100, 30)
    yield* terminal.write("size\r")
    yield* waitForTerminal(terminal, s => s.lines.some(line => line.includes("SIZE:100x30")), "receive native resize")
    yield* terminal.write("paint\r")
    const rendered = yield* waitForTerminal(terminal, s => s.lines.some(line => line.includes("AFTER colored 漢字")), "render updated output")
    expect(rendered.lines.join("\n")).not.toContain("BEFORE")
    expect(rendered.columns).toBe(100)
    yield* terminal.write("alternate\r")
    yield* waitForTerminal(terminal, s => s.alternate && s.lines.some(line => line.includes("ALTERNATE")), "enter the alternate screen")
    yield* terminal.write("normal\r")
    yield* waitForTerminal(terminal, s => !s.alternate && s.lines.some(line => line.includes("AFTER")), "restore the normal screen")
    yield* terminal.write("\u0003")
    yield* waitForTerminal(terminal, s => s.lines.some(line => line.includes("INTERRUPTED")), "handle interrupt input")
    yield* terminal.write("again\r")
    yield* waitForTerminal(terminal, s => s.lines.some(line => line.includes("RECOVERED")), "remain usable after interruption")
    yield* terminal.write("\u0004")
    expect((yield* terminal.exited).code).toBe(0)
  }))
  expect(cleanup).toEqual([])
  const screen = yield* fs.readFileString(join(root, "evidence", "terminal-screen.json")).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(TerminalScreen))))
  expect(screen.lines.join("\n")).toContain("EXITED")
  expect(yield* fs.readFileString(join(root, "evidence", "terminal-output.txt"))).toContain("\u001b[31m")
})).pipe(Effect.timeout("35 seconds"), Effect.provide(Layer.merge(BunContext.layer, NativeTerminalDriver.pipe(Layer.provide(BunContext.layer)))))), 45000)

test.skipIf(!runtime)("terminal ownership cleans up a child that ignores graceful termination", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-terminal-cleanup-" }), script = join(root, "fixture.cjs")
  yield* fs.writeFileString(script, `process.on('SIGTERM',()=>{});setInterval(()=>{},1000);process.stdout.write('READY\\r\\n')`)
  const cleanup: string[] = []
  const pid = yield* Effect.scoped(Effect.gen(function* () {
    const terminal = yield* (yield* TerminalDriver).start(config(root, script, "evidence"), message => { cleanup.push(message) })
    yield* waitForTerminal(terminal, s => s.lines.some(line => line.includes("READY")), "start the cleanup fixture")
    return terminal.pid
  }))
  expect(cleanup).toEqual([])
  expect(() => process.kill(pid, 0)).toThrow()
})).pipe(Effect.timeout("35 seconds"), Effect.provide(Layer.merge(BunContext.layer, NativeTerminalDriver.pipe(Layer.provide(BunContext.layer)))))), 45000)


test.skipIf(!runtime)("cancelling a terminal session reaps its owned child", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-terminal-cancel-" }), script = join(root, "fixture.cjs")
  yield* fs.writeFileString(script, `process.on('SIGTERM',()=>{});setInterval(()=>{},1000);process.stdout.write('READY\\r\\n')`)
  const cleanup: string[] = [], started = yield* Deferred.make<number>()
  const fiber = yield* Effect.scoped(Effect.gen(function* () {
    const terminal = yield* (yield* TerminalDriver).start(config(root, script, "evidence"), message => { cleanup.push(message) })
    yield* waitForTerminal(terminal, s => s.lines.some(line => line.includes("READY")), "start the cancellation fixture")
    yield* Deferred.succeed(started, terminal.pid)
    return yield* Effect.never
  })).pipe(Effect.fork)
  const pid = yield* Deferred.await(started)
  yield* Fiber.interrupt(fiber)
  expect(cleanup).toEqual([])
  expect(() => process.kill(pid, 0)).toThrow()
})).pipe(Effect.timeout("35 seconds"), Effect.provide(Layer.merge(BunContext.layer, NativeTerminalDriver.pipe(Layer.provide(BunContext.layer)))))), 45000)

test.skipIf(!runtime)("missing native executables cannot qualify as a terminal session", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-terminal-missing-" })
  const cleanup: string[] = []
  const result = yield* Effect.scoped(TerminalDriver.pipe(Effect.flatMap(driver => driver.start({ ...config(root, "unused", "evidence"), executable: join(root, "missing") }, message => { cleanup.push(message) })))).pipe(Effect.either)
  expect(result._tag).toBe("Left")
  expect(cleanup).toEqual([])
})).pipe(Effect.timeout("35 seconds"), Effect.provide(Layer.merge(BunContext.layer, NativeTerminalDriver.pipe(Layer.provide(BunContext.layer)))))), 45000)


test.skipIf(!runtime)("excessive terminal output fails and cleans up the native child", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-terminal-bound-" }), script = join(root, "fixture.cjs")
  yield* fs.writeFileString(script, `process.stdin.setRawMode(true);process.stdin.resume();process.stdout.write('READY\\r\\n');process.stdin.once('data',()=>{process.stdout.write('x'.repeat(17*1024*1024))});setInterval(()=>{},1000)`)
  const cleanup: string[] = []
  let pid = 0
  const result = yield* Effect.scoped(Effect.gen(function* () {
    const terminal = yield* (yield* TerminalDriver).start(config(root, script, "evidence"), message => { cleanup.push(message) })
    pid = terminal.pid
    yield* waitForTerminal(terminal, s => s.lines.some(line => line.includes("READY")), "start the output-bound fixture")
    yield* terminal.write("x")
    yield* terminal.exited
  })).pipe(Effect.either)
  expect(result._tag).toBe("Left")
  expect(() => process.kill(pid, 0)).toThrow()
  expect(Number((yield* fs.stat(join(root, "evidence", "terminal-output.txt"))).size)).toBeLessThanOrEqual(16 * 1024 * 1024)
  expect(cleanup).toEqual([])
  expect(yield* fs.exists(join(root, "evidence", "terminal-screen.json"))).toBe(true)
  expect(yield* fs.exists(join(root, "evidence", "failure-screen.json"))).toBe(true)
})).pipe(Effect.timeout("35 seconds"), Effect.provide(Layer.merge(BunContext.layer, NativeTerminalDriver.pipe(Layer.provide(BunContext.layer)))))), 45000)
