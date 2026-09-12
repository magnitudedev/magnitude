import { win32 } from "node:path"
import { Effect, Schema } from "effect"
export interface WindowsChildCommand {
  readonly executable: string
  readonly arguments: ReadonlyArray<string>
  readonly environment: Readonly<Record<string, string | undefined>>
}

const text = Schema.String.pipe(Schema.filter(value => !value.includes("\0")))
const executable = text.pipe(Schema.filter(value => win32.isAbsolute(value) && win32.parse(value).root.length > 1 && !value.includes('"')))
const environmentKey = text.pipe(Schema.filter(value => value.length > 0 && (!value.includes("=") || /^=[a-z]:$/i.test(value))))
const commandSchema = Schema.Struct({
  executable,
  arguments: Schema.Array(text),
  environment: Schema.Record({ key: environmentKey, value: text }),
})
export class WindowsCommandInvalid extends Schema.TaggedError<WindowsCommandInvalid>()("WindowsCommandInvalid", { message: Schema.String }) {}
export const WindowsCommandLine = Schema.String.pipe(Schema.maxLength(32766), Schema.brand("WindowsCommandLine"))
export type WindowsCommandLine = typeof WindowsCommandLine.Type
export const WindowsEnvironmentBlock = Schema.String.pipe(Schema.maxLength(1048576), Schema.brand("WindowsEnvironmentBlock"))
export type WindowsEnvironmentBlock = typeof WindowsEnvironmentBlock.Type

/** CRT argv quoting for the known Node/Bun/Rust executable; never invokes a command shell. */
const quote = (argument: string) => '"' + argument.replace(/(\\*)"/g, '$1$1\\"').replace(/(\\+)$/g, '$1$1') + '"'
export const encodeWindowsCommand = (input: WindowsChildCommand) => Effect.gen(function* () {
  const command = yield* Schema.decodeUnknown(commandSchema, { onExcessProperty: "error" })({ ...input, environment: Object.fromEntries(
    Object.entries(input.environment).filter((entry): entry is [string, string] => entry[1] !== undefined),
  ) }).pipe(Effect.mapError(() => new WindowsCommandInvalid({ message: "The Windows child command contains an invalid executable, argument or environment entry." })))
  const commandLine = yield* Schema.decodeUnknown(WindowsCommandLine)([command.executable, ...command.arguments].map(quote).join(" ")).pipe(
    Effect.mapError(() => new WindowsCommandInvalid({ message: "The Windows child command exceeds the native command-line limit." })),
  )
  // The native boundary sorts using Windows ordinal comparison and rejects duplicate names.
  const environment = yield* Schema.decodeUnknown(WindowsEnvironmentBlock)(Object.entries(command.environment).map(([key, value]) => `${key}=${value}`).join("\0") + "\0\0").pipe(
    Effect.mapError(() => new WindowsCommandInvalid({ message: "The Windows child environment exceeds the native buffer limit." })),
  )
  return { executable: command.executable, commandLine, environment }
})
