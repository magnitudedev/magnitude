import { Cause, Data, Effect, Schema, type ParseResult } from "effect"
import stringWidth from "string-width"

const messageOf = (error: unknown): string => {
  if (typeof error === "object" && error !== null) {
    const candidate = error as { readonly message?: unknown; readonly reason?: unknown }
    if (Object.hasOwn(candidate, "message") && typeof candidate.message === "string") return candidate.message
    if (typeof candidate.reason === "string") return candidate.reason
    if (typeof candidate.message === "string") return candidate.message
  }
  return String(error)
}

/** A stable machine code for a failure: the product code it carries, else its tag in snake case. */
export const errorCode = (error: unknown): string => {
  if (typeof error === "object" && error !== null) {
    const candidate = error as { readonly code?: unknown; readonly _tag?: unknown }
    if (typeof candidate.code === "string" && candidate.code.length > 0) return candidate.code
    if (typeof candidate._tag === "string" && candidate._tag.length > 0) {
      return candidate._tag.replace(/([a-z0-9])([A-Z])/g, "$1_$2").toLowerCase()
    }
  }
  return "command_failed"
}

/** What a failed command writes to stderr: its product message, or under `--json` one error document. */
export const renderCommandError = (error: unknown, json: boolean): string => json
  ? `${JSON.stringify({ error: { message: messageOf(error), code: errorCode(error) } }, null, 2)}\n`
  : `${messageOf(error)}\n`

class CommandOutputFailed extends Data.TaggedError("CommandOutputFailed")<{
  readonly message: string
}> {}

/** How a command's result becomes the one machine-readable document `--json` prints. */
export interface JsonDocument<A> {
  readonly encode: (result: A) => Effect.Effect<unknown, ParseResult.ParseError>
}

export const jsonDocument = <A, J, I>(
  schema: Schema.Schema<J, I, never>,
  project: (result: A) => J,
): JsonDocument<A> => ({ encode: (result) => Schema.encode(schema)(project(result)) })

export interface JsonOption {
  readonly json?: boolean
}

/** The document to print when `--json` was given, or nothing for text output. */
export const asJson = <A>(options: JsonOption, document: JsonDocument<A>): JsonDocument<A> | undefined =>
  options.json === true ? document : undefined

export const runCommand = <A>(options: {
  readonly effect: Effect.Effect<A, unknown, never>
  readonly render: (result: A) => string
  readonly json?: JsonDocument<A>
}): Promise<void> => Effect.runPromise(options.effect.pipe(
  Effect.flatMap((result) => options.json === undefined
    ? Effect.succeed(options.render(result))
    : options.json.encode(result).pipe(Effect.map((document) => `${JSON.stringify(document, null, 2)}\n`))),
  Effect.flatMap((output) => Effect.try({
    try: () => process.stdout.write(output),
    catch: (error) => new CommandOutputFailed({ message: messageOf(error) }),
  })),
  Effect.catchAllCause((cause) => Effect.sync(() => {
    process.stderr.write(renderCommandError(Cause.squash(cause), options.json !== undefined))
    process.exitCode = 1
  })),
  Effect.asVoid,
))

export const ensureTrailingNewline = (value: string): string =>
  `${value.replace(/\n+$/, "")}\n`

export interface TableColumn<Row> {
  readonly heading: string
  readonly value: (row: Row) => string
}

export const renderTable = <Row>(
  rows: readonly Row[],
  columns: readonly TableColumn<Row>[],
  terminalWidth = process.stdout.isTTY ? process.stdout.columns ?? 120 : Number.POSITIVE_INFINITY,
): string => {
  const values = rows.map((row) => columns.map(({ value }) => value(row)))
  const widths = columns.map(({ heading }, index) => Math.max(
    stringWidth(heading),
    ...values.map((row) => stringWidth(row[index]!)),
  ))
  const tableWidth = widths.reduce((total, width) => total + width, 0) + (columns.length - 1) * 2
  if (tableWidth > terminalWidth) {
    const labelWidth = Math.max(...columns.map(({ heading }) => heading.length))
    return ensureTrailingNewline(rows.map((row) => columns.map(({ heading, value }) =>
      `  ${heading.padEnd(labelWidth)}  ${value(row)}`
    ).join("\n")).join("\n\n"))
  }
  const line = (cells: readonly string[]) => cells.map((cell, index) =>
    index === cells.length - 1 ? cell : `${cell}${" ".repeat(widths[index]! - stringWidth(cell))}`
  ).join("  ")
  return ensureTrailingNewline([
    line(columns.map(({ heading }) => heading)),
    ...values.map(line),
  ].join("\n"))
}

export const renderFields = (fields: readonly (readonly [string, string])[]): string => {
  const width = Math.max(...fields.map(([label]) => label.length), 0)
  return fields.map(([label, value]) => `  ${label.padEnd(width)}  ${value}`).join("\n")
}
