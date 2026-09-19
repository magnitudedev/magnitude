import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer, Schema } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { Candidate } from "../src/candidate"
import { targets } from "../src/catalog"
import { InstalledApplication } from "../src/installer"
import { ProcessExecutor } from "../src/process"
import { verifyNativeRemoval } from "../src/suites/uninstall"

const fixtures = [
  { format: "deb", stdout: "other\tinstalled\n", code: 0, passes: true },
  { format: "deb", stdout: "magnitude-desktop\tconfig-files\n", code: 0, passes: true },
  { format: "deb", stdout: "magnitude-desktop:amd64\tinstalled\n", code: 0, passes: false },
  { format: "deb", stdout: "magnitude-desktop\thalf-installed\n", code: 0, passes: false },
  { format: "rpm", stdout: "other\n", code: 0, passes: true },
  { format: "rpm", stdout: "magnitude-desktop\n", code: 0, passes: false },
  { format: "rpm", stdout: "", code: 1, passes: false },
  { format: "exe", stdout: '{"registered":false,"shortcut":false,"cliPath":false}', code: 0, passes: true },
  { format: "exe", stdout: '{"registered":true,"shortcut":false,"cliPath":false}', code: 0, passes: false },
  { format: "exe", stdout: '{"registered":false,"shortcut":true,"cliPath":false}', code: 0, passes: false },
  { format: "exe", stdout: '{"registered":false,"shortcut":false,"cliPath":true}', code: 0, passes: false },
  { format: "exe", stdout: '{}', code: 0, passes: false },
] as const
for (const [index, fixture] of fixtures.entries()) test(`registration removal ${index}: ${fixture.format}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-registration-" })
  const target = targets.find(target => target.packageFormat === fixture.format)!
  const candidate = yield* Schema.decodeUnknown(Candidate)({ version: "0.1.3", target, path: join(root, `candidate.${fixture.format}`),
    artifact: { id: `desktop-${target.artifactHost}`, kind: "desktop", host: target.artifactHost, filename: `Magnitude.${fixture.format}`, bytes: 1, sha256: "a".repeat(64) } })
  const application = InstalledApplication.make({ candidate, root: join(root, "app"), executable: join(root, "app", "exe"), cli: join(root, "cli"), packageVersion: "0.1.3" })
  const result = yield* verifyNativeRemoval(application, {}).pipe(Effect.provide(Layer.succeed(ProcessExecutor, { run: spec => {
    expect(spec.inheritEnv).toBe(false)
    if (fixture.format === "exe") {
      expect(spec.args.at(-1)).toContain("HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\MagnitudeDesktop")
      expect(spec.env.LAB_REMOVED_CLI_DIRECTORY).toBe(application.root + "\\resources")
    }
    return Effect.succeed({ exitCode: fixture.code, stdout: fixture.stdout, stderr: "" })
  } })), Effect.either)
  expect(result._tag).toBe(fixture.passes ? "Right" : "Left")
})).pipe(Effect.provide(BunContext.layer))))
