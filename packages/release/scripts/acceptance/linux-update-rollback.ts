import { Command, FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Effect, Schema } from "effect"
import { join, resolve } from "node:path"
import { renderLinuxMaintainerScripts } from "../build/linux-maintainer-scripts"

class AcceptanceFailed extends Schema.TaggedError<AcceptanceFailed>()("AcceptanceFailed", { message: Schema.String }) {}
const root = resolve(process.cwd())
const name = "magnitude-update-rollback-fixture"
const marker = "/var/lib/magnitude-desktop/installing"
const run = Effect.scoped(Effect.gen(function* () {
  if (process.platform !== "linux") return yield* new AcceptanceFailed({ message: "Run this fixture on an isolated Debian-family runner." })
  const fs = yield* FileSystem.FileSystem
  const installed = yield* Command.make("dpkg-query", "-W", "-f=${Status}", "magnitude-desktop").pipe(Command.string, Effect.option)
  if (installed._tag === "Some" && installed.value.includes("installed")) return yield* new AcceptanceFailed({ message: "Do not run against an installed desktop." })
  if (yield* fs.exists(marker)) return yield* new AcceptanceFailed({ message: "An existing installation transaction must not be disturbed." })
  const directory = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-rollback-" })
  const begin = yield* fs.readFileString(join(root, "packages/release/resources/linux/installation-begin.sh"))
  const end = yield* fs.readFileString(join(root, "packages/release/resources/linux/installation-end.sh"))
  const scripts = renderLinuxMaintainerScripts(begin, end)
  const install = (path: string) => Command.make("sudo", "dpkg", "--install", path).pipe(Command.stdout("inherit"), Command.stderr("inherit"), Command.exitCode)
  for (const version of ["1.0.0", "2.0.0"]) {
    const stage = join(directory, version)
    yield* fs.makeDirectory(join(stage, "DEBIAN"), { recursive: true })
    yield* fs.writeFileString(join(stage, "DEBIAN/control"), `Package: ${name}\nVersion: ${version}\nArchitecture: all\nMaintainer: Magnitude Acceptance\nDescription: Isolated installation rollback fixture\n`)
    for (const [script, body] of Object.entries(scripts)) {
      const path = join(stage, "DEBIAN", script)
      yield* fs.writeFileString(path, `#!/bin/sh\nset -eu\n${body}\n${version === "2.0.0" && script === "preinst" ? "exit 1\n" : ""}`)
      yield* fs.chmod(path, 0o755)
    }
    if ((yield* Command.make("dpkg-deb", "--root-owner-group", "--build", stage, `${stage}.deb`).pipe(Command.exitCode)) !== 0) {
      return yield* new AcceptanceFailed({ message: "Could not build the rollback fixture." })
    }
  }
  yield* Effect.gen(function* () {
    if ((yield* install(join(directory, "1.0.0.deb"))) !== 0) return yield* new AcceptanceFailed({ message: "Baseline installation failed." })
    if ((yield* install(join(directory, "2.0.0.deb"))) === 0) return yield* new AcceptanceFailed({ message: "Injected upgrade failure was not observed." })
    const status = yield* Command.make("dpkg-query", "-W", "-f=${Version} ${Status}", name).pipe(Command.string)
    if (status !== "1.0.0 install ok installed" || (yield* fs.exists(marker))) {
      return yield* new AcceptanceFailed({ message: "Dpkg rollback did not restore admission to the old installation." })
    }
    if ((yield* Command.make("flock", "--shared", "--nonblock", "/var/lib/magnitude-desktop/installation.lock", "true").pipe(Command.exitCode)) !== 0) {
      return yield* new AcceptanceFailed({ message: "Rollback retained the installation lease." })
    }
    yield* Effect.log("PASS real dpkg rollback restores old package state and application admission")
  }).pipe(Effect.ensuring(Command.make("sudo", "dpkg", "--purge", name).pipe(Command.exitCode, Effect.ignore)))
}))
BunRuntime.runMain(run.pipe(Effect.provide(BunContext.layer)))
