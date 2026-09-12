import { Effect } from "effect"
import { describe, expect, it } from "vitest"
import { encodeWindowsCommand } from "./windows-command"

const executable = String.raw`C:\Program Files\Magnitude\magnitude-service.exe`
describe("Windows service command encoding", () => {
  it("preserves empty, quoted, Unicode and shell-looking arguments without invoking a shell", async () => {
    const result = await Effect.runPromise(encodeWindowsCommand({ executable,
      arguments: ["", "two words", 'say "hello"', "模型🙂", "& echo %PATH%", "line\nbreak", "ends\\"], environment: {},
    }))
    expect(result.commandLine).toBe('"C:\\Program Files\\Magnitude\\magnitude-service.exe" "" "two words" "say \\"hello\\"" "模型🙂" "& echo %PATH%" "line\nbreak" "ends\\\\"')
    expect(result.environment).toBe("\0\0")
  })

  it("preserves backslashes before literal quotes", async () => {
    const result = await Effect.runPromise(encodeWindowsCommand({ executable, arguments: ['a\\"b', 'a\\\\"b'], environment: {} }))
    expect(result.commandLine).toBe('"' + executable + '" "a' + "\\".repeat(3) + '"b" "a' + "\\".repeat(5) + '"b"')
  })

  it("omits absent values and preserves empty values, drive state and equals signs in values", async () => {
    const result = await Effect.runPromise(encodeWindowsCommand({ executable, arguments: [], environment: { OMIT: undefined, EMPTY: "", "=C:": "C:\\work", VALUE: "a=b" } }))
    expect(result.environment).toBe("EMPTY=\0=C:=C:\\work\0VALUE=a=b\0\0")
  })

  it.each([
    { executable: "relative.exe", arguments: [], environment: {} },
    { executable: "\\drive-relative.exe", arguments: [], environment: {} },
    { executable: 'C:\\bad"name.exe', arguments: [], environment: {} },
    { executable, arguments: ["embedded\0nul"], environment: {} },
    { executable, arguments: [], environment: { "BAD=KEY": "value" } },
    { executable, arguments: [], environment: { "": "value" } },
    { executable, arguments: [], environment: { "=BAD": "value" } },
    { executable, arguments: [], environment: { VALID: "embedded\0nul" } },
  ])("rejects malformed input %# instead of silently changing it", async command => {
    const result = await Effect.runPromise(Effect.either(encodeWindowsCommand(command)))
    expect(result._tag).toBe("Left")
  })

  it("enforces the final encoded command and native environment buffer limits", async () => {
    for (const command of [
      { executable, arguments: ["\\".repeat(32700)], environment: {} },
      { executable, arguments: [], environment: { HUGE: "x".repeat(1048576) } },
    ]) expect((await Effect.runPromise(Effect.either(encodeWindowsCommand(command))))._tag).toBe("Left")
  })
})
