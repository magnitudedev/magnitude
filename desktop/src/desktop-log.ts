import { NodeContext } from "@effect/platform-node"
import { Effect, Layer, Logger } from "effect"
import { join } from "node:path"
import { openLogFile } from "@magnitudedev/utils/log-file"

/** The desktop's own log lines, kept beside the service and inference logs in the data directory. */
export const desktopLogLayer = (dataDirectory: string) => Logger.addScoped(
  openLogFile(join(dataDirectory, "logs", "desktop.log"), 10 * 1024 * 1024).pipe(Effect.map(file =>
    Logger.logfmtLogger.pipe(Logger.map(line => Effect.runSync(file.append(`${line}\n`)))))),
).pipe(Layer.provide(NodeContext.layer))
