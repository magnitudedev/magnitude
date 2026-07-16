import { Cause, Layer, Logger } from "effect"
import { addEphemeralLogMessage } from "@magnitudedev/client-common"
import { TracingLayer } from "@magnitudedev/sdk"

const stringifyMessagePart = (value: unknown): string => {
  if (typeof value === "string") return value
  if (value instanceof Error) return value.message
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

const formatToastMessage = (message: unknown, cause: Cause.Cause<unknown>): string => {
  const text = (Array.isArray(message) ? message : [message])
    .map(stringifyMessagePart)
    .filter(Boolean)
    .join(" ")

  if (text) return text
  return Cause.isEmpty(cause) ? "An unexpected error occurred" : Cause.pretty(cause)
}

export interface MakeCliEffectLoggingLayerOptions {
  readonly debug: boolean
}

/**
 * CLI Effect logging fan-out. OTLP remains the structured log destination;
 * errors are also surfaced in the TUI, as are warnings in debug mode.
 */
export const makeCliEffectLoggingLayer = (
  options: MakeCliEffectLoggingLayerOptions,
): Layer.Layer<never, never, never> => {
  const toastLogger = Logger.make(({ logLevel, message, cause }) => {
    const tone = logLevel.label === "FATAL" || logLevel.label === "ERROR"
      ? "error"
      : logLevel.label === "WARN" && options.debug
        ? "warning"
        : null

    if (tone) {
      addEphemeralLogMessage(formatToastMessage(message, cause), tone)
    }
  })

  return Layer.mergeAll(
    TracingLayer,
    Logger.replace(Logger.defaultLogger, Logger.none),
    Logger.add(toastLogger),
  )
}
