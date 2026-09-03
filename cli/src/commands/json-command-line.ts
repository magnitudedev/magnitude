import type { CommandUnknownOpts } from "@commander-js/extra-typings"
import type { JsonCommandName } from "./output"

const commandName = (args: readonly string[]): JsonCommandName | undefined => {
  if (args[0] !== "models") return undefined
  switch (args[1]) {
    case "status": return "models.status"
    case "load": return "models.load"
    case "stop": return "models.stop"
    default: return undefined
  }
}

export const requestedJsonCommand = (args: readonly string[]): JsonCommandName | undefined => {
  const separator = args.indexOf("--")
  const options = separator === -1 ? args : args.slice(0, separator)
  return options.includes("--json") && !options.includes("--help") && !options.includes("-h")
    ? commandName(args)
    : undefined
}

const parserMessage = (error: unknown): string => {
  const message = error instanceof Error ? error.message : String(error)
  return message.replace(/^error:\s*/u, "").replace(/\n+$/u, "")
}

export const parseJsonCommand = async (
  program: CommandUnknownOpts,
  args: readonly string[],
  command: JsonCommandName,
): Promise<void> => {
  const suppressParserError = (candidate: CommandUnknownOpts): void => {
    candidate.exitOverride()
    candidate.configureOutput({ writeErr: () => undefined })
    for (const child of candidate.commands) suppressParserError(child)
  }
  suppressParserError(program)
  try {
    await program.parseAsync(args, { from: "user" })
  } catch (error) {
    const { renderJsonCommandFailure } = await import("./output")
    process.stderr.write(renderJsonCommandFailure(command, parserMessage(error)))
    process.exitCode = typeof error === "object"
      && error !== null
      && "exitCode" in error
      && typeof error.exitCode === "number"
      ? error.exitCode
      : 1
  }
}
