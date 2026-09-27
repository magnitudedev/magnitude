import { createRequire } from "node:module"
import { spawn } from "node:child_process"
import { existsSync, mkdirSync } from "node:fs"
import { resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { Effect, Schema } from "effect"

class NativeBuildFailed extends Schema.TaggedError<NativeBuildFailed>()("NativeBuildFailed", {
  message: Schema.String,
}) {}

/** The desktop tray addon: macOS only, where the tray's live model row is a native view. */
const build = Effect.gen(function* () {
  if (process.platform !== "darwin") return
  const root = fileURLToPath(new URL("../", import.meta.url))
  const headers: string = process.env.MAGNITUDE_NODE_HEADERS ?? createRequire(import.meta.url)("node-api-headers").include_dir
  if (!headers || !existsSync(resolve(headers, "node_api.h"))) {
    return yield* new NativeBuildFailed({ message: "Set MAGNITUDE_NODE_HEADERS to a Node-API include directory containing node_api.h" })
  }
  const output = resolve(root, "dist/native", `${process.platform}-${process.arch}`, "tray-status.node")
  yield* Effect.try({
    try: () => mkdirSync(resolve(output, ".."), { recursive: true }),
    catch: error => new NativeBuildFailed({ message: String(error) }),
  })
  const args = ["-fobjc-arc", "-DNAPI_VERSION=8", "-O2", "-Wall", "-Wextra", "-Werror", "-shared",
    "-undefined", "dynamic_lookup", "-mmacosx-version-min=13.0", "-framework", "AppKit",
    "-I", headers, resolve(root, "native/tray-status.m"), "-o", output]
  yield* Effect.async<void, NativeBuildFailed>(resume => {
    const child = spawn(process.env.CC ?? "cc", args, { stdio: "inherit" })
    child.once("error", error => resume(Effect.fail(new NativeBuildFailed({ message: error.message }))))
    child.once("exit", code => resume(code === 0 ? Effect.void : Effect.fail(new NativeBuildFailed({ message: `Compiler exited ${code}` }))))
    return Effect.sync(() => { if (child.exitCode === null) child.kill() })
  })
  yield* Effect.log(`Built ${output}`)
})
Effect.runPromise(build).catch(error => { console.error(String(error)); process.exitCode = 1 })
