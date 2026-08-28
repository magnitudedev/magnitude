import { FetchHttpClient } from "@effect/platform"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as Path from "@effect/platform/Path"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import {
  type ServerServiceError,
  startServer,
  stopServer,
} from "../server/service"

type ServerRequirements =
  | FileSystem.FileSystem
  | Path.Path
  | CommandExecutor.CommandExecutor
  | HttpClient.HttpClient

const runCommand = (
  effect: Effect.Effect<void, ServerServiceError, ServerRequirements>,
  success: string,
) => Effect.runPromise(effect.pipe(
  Effect.provide([BunContext.layer, FetchHttpClient.layer]),
  Effect.tap(() => Effect.sync(() => process.stdout.write(`${success}\n`))),
  Effect.catchAll((error) => Effect.sync(() => {
    process.stderr.write(`${error.message}\n`)
    process.exitCode = 1
  })),
))

export const runServerStart = () =>
  runCommand(startServer, "Magnitude server is ready at 127.0.0.1:10100.")

export const runServerStop = () =>
  runCommand(stopServer, "Magnitude server stopped.")
