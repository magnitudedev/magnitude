import { Data, Effect } from "effect"
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

class CommandOutputFailed extends Data.TaggedError("CommandOutputFailed")<{
  readonly message: string
}> {}

export const runCommand = <A>(options: {
  readonly effect: Effect.Effect<A, unknown, never>
  readonly render: (result: A) => string
}): Promise<void> => Effect.runPromise(options.effect.pipe(
  Effect.tap((result) => Effect.try({
    try: () => process.stdout.write(options.render(result)),
    catch: (error) => new CommandOutputFailed({ message: messageOf(error) }),
  })),
  Effect.catchAll((error) => Effect.sync(() => {
    process.stderr.write(`${messageOf(error)}\n`)
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
