import { createRequire } from "node:module"
import { spawn } from "node:child_process"
import { existsSync, mkdirSync } from "node:fs"
import { resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { Effect, Schema } from "effect"

class NativeBuildFailed extends Schema.TaggedError<NativeBuildFailed>()("NativeBuildFailed", {
  message: Schema.String,
}) {}

const build = Effect.gen(function* () {
  const root = fileURLToPath(new URL("../", import.meta.url))
  const headers: string = process.env.MAGNITUDE_NODE_HEADERS ?? createRequire(import.meta.url)("node-api-headers").include_dir
  if (!headers || !existsSync(resolve(headers, "node_api.h"))) {
    return yield* new NativeBuildFailed({ message: "Set MAGNITUDE_NODE_HEADERS to a Node-API include directory containing node_api.h" })
  }
  const output = resolve(root, "dist/native", `${process.platform}-${process.arch}`, "desktop-host.node")
  yield* Effect.try({
    try: () => mkdirSync(resolve(output, ".."), { recursive: true }),
    catch: error => new NativeBuildFailed({ message: String(error) }),
  })
  if (process.platform === "win32") {
    const library = process.env.MAGNITUDE_NODE_LIBRARY
    if (process.arch !== "x64" || !library || !existsSync(library)) {
      return yield* new NativeBuildFailed({ message: "Windows x64 native build requires Visual Studio C++ tools and MAGNITUDE_NODE_LIBRARY pointing to a verified x64 node.lib" })
    }
    yield* Effect.async<void, NativeBuildFailed>(resume => {
      const child = spawn("pwsh", ["-NoProfile", "-File", resolve(root, "scripts/build-windows-native.ps1"),
        "-Headers", headers, "-NodeLibrary", library, "-Output", output], { stdio: "inherit" })
      child.once("error", error => resume(Effect.fail(new NativeBuildFailed({ message: error.message }))))
      child.once("exit", code => resume(code === 0 ? Effect.void : Effect.fail(new NativeBuildFailed({ message: `Windows native build exited ${code}` }))))
      return Effect.sync(() => { if (child.exitCode === null) child.kill() })
    })
    yield* Effect.log(`Built ${output}`)
    return
  }
  const args = ["-std=c11", "-D_GNU_SOURCE", "-DNAPI_VERSION=8", "-O2", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared", "-pthread",
    ...(process.platform === "darwin" ? ["-undefined", "dynamic_lookup", "-mmacosx-version-min=13.0"] : []),
    "-I", headers, resolve(root, "native/desktop-host.c"), resolve(root, "native/application-memory.c"), resolve(root, "native/machine-identity.c"), "-o", output]
  yield* Effect.async<void, NativeBuildFailed>(resume => {
    const child = spawn(process.env.CC ?? "cc", args, { stdio: "inherit" })
    child.once("error", error => resume(Effect.fail(new NativeBuildFailed({ message: error.message }))))
    child.once("exit", code => resume(code === 0 ? Effect.void : Effect.fail(new NativeBuildFailed({ message: `Compiler exited ${code}` }))))
    return Effect.sync(() => { if (child.exitCode === null) child.kill() })
  })
  const helper = resolve(output, "../magnitude-command")
  yield* Effect.async<void, NativeBuildFailed>(resume => {
    const child = spawn(process.env.CC ?? "cc", ["-std=c11", "-D_GNU_SOURCE", "-O2", "-Wall", "-Wextra", "-Werror",
      ...(process.platform === "darwin" ? ["-mmacosx-version-min=13.0"] : []),
      resolve(root, "native/owned-command.c"), "-o", helper], { stdio: "inherit" })
    child.once("error", error => resume(Effect.fail(new NativeBuildFailed({ message: error.message }))))
    child.once("exit", code => resume(code === 0 ? Effect.void : Effect.fail(new NativeBuildFailed({ message: `Command helper compiler exited ${code}` }))))
    return Effect.sync(() => { if (child.exitCode === null) child.kill() })
  })
  yield* Effect.log(`Built ${output}`)
})
Effect.runPromise(build).catch(error => { console.error(String(error)); process.exitCode = 1 })
