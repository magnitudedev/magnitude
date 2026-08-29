import { Effect, Schema } from "effect"

export type OutputMode = "human" | "json"

let outputMode: OutputMode = "human"

export const setOutputMode = (json: boolean): void => {
  outputMode = json ? "json" : "human"
}

export const currentOutputMode = (): OutputMode => outputMode

export const CliErrorSchema = Schema.Struct({
  error: Schema.Struct({
    code: Schema.String,
    message: Schema.String,
    retryable: Schema.Boolean,
  }),
})

export type CliErrorDocument = typeof CliErrorSchema.Type

const messageOf = (error: unknown): string => {
  if (typeof error === "object" && error !== null) {
    if ("reason" in error && typeof error.reason === "string") return error.reason
    if ("message" in error && typeof error.message === "string") return error.message
  }
  return String(error)
}

export const errorDocument = (error: unknown): CliErrorDocument => ({
  error: {
    code: typeof error === "object" && error !== null && "code" in error
      && typeof error.code === "string" ? error.code : "command_failed",
    message: messageOf(error),
    retryable: typeof error === "object" && error !== null && "retryable" in error
      && typeof error.retryable === "boolean" ? error.retryable : false,
  },
})

export const encodeErrorDocument = (error: unknown): typeof CliErrorSchema.Encoded =>
  Schema.encodeSync(CliErrorSchema)(errorDocument(error))

const writeJson = (stream: NodeJS.WriteStream, value: unknown): void => {
  stream.write(`${JSON.stringify(value)}\n`)
}

export const runCommand = <S, I, A extends S>(options: {
  readonly effect: Effect.Effect<A, unknown, never>
  readonly schema: Schema.Schema<S, I, never>
  readonly render: (result: A) => string
}): Promise<void> => {
  const encoded = options.effect.pipe(
    Effect.flatMap((result) => Schema.encode(options.schema)(result).pipe(
      Effect.map((document) => ({ result, document })),
    ),
  ))
  return Effect.runPromise(encoded.pipe(
    Effect.tap(({ result, document }) => Effect.sync(() => {
      if (outputMode === "json") writeJson(process.stdout, document)
      else process.stdout.write(options.render(result))
    })),
    Effect.catchAll((error) => Effect.sync(() => {
      if (outputMode === "json") writeJson(process.stderr, encodeErrorDocument(error))
      else process.stderr.write(`${messageOf(error)}\n`)
      process.exitCode = 1
    })),
    Effect.asVoid,
  ))
}

export const ensureTrailingNewline = (value: string): string =>
  `${value.replace(/\n+$/, "")}\n`
