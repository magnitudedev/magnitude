import { Command, FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Effect, Schema } from "effect"
import { createHash, generateKeyPairSync } from "node:crypto"
import { join, resolve } from "node:path"
import { signUpdateRelease } from "../../src/hosted-update/release"
import { InstallationOffer } from "../../src/hosted-update/installation-offer"

class AcceptanceFailed extends Schema.TaggedError<AcceptanceFailed>()("AcceptanceFailed", { message: Schema.String }) {}
const run = Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude bootstrap ' " })
  const binary = join(root, process.platform === "win32" ? "verifier.exe" : "verifier")
  const keys = yield* Effect.sync(() => generateKeyPairSync("ed25519"))
  const publicKey = keys.publicKey.export({ type: "spki", format: "pem" }).toString()
  const build = yield* Command.make(process.execPath, "build", resolve(import.meta.dir, "../../../../cli/src/index.ts"), "--compile",
    `--outfile=${binary}`, "--external", "electron", "--external", "chromium-bidi", "--define", `MAGNITUDE_BOOTSTRAP_PUBLISHER_KEY=${JSON.stringify(publicKey)}`).pipe(
    Command.stdout("inherit"), Command.stderr("inherit"), Command.exitCode)
  if (build !== 0) return yield* new AcceptanceFailed({ message: "Bootstrap verifier compilation failed" })
  const artifact = join(root, "installer.exe"), offerPath = join(root, "offer.json"), profile = join(root, "profile")
  const bytes = Buffer.from("bootstrap installer fixture")
  yield* fs.writeFile(artifact, bytes)
  const release = yield* signUpdateRelease({ version: "0.1.6", bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") },
    { os: "windows", arch: "x64", package: "windows-exe" }, keys.privateKey)
  yield* fs.writeFileString(offerPath, yield* Schema.encode(Schema.parseJson(InstallationOffer))({ release,
    download: "https://github.com/magnitudedev/magnitude/releases/download/test/installer.exe" }))
  const verify = Command.make(binary, "_verify-windows-installation", offerPath, artifact, "stable").pipe(
    Command.env({ MAGNITUDE_DEV_DATA_DIR: profile, MAGNITUDE_BOOTSTRAP_PUBLISHER_KEY: "not-a-runtime-trust-override" }),
    Command.stdout("inherit"), Command.stderr("inherit"), Command.exitCode)
  if ((yield* verify) !== 0) return yield* new AcceptanceFailed({ message: "The compiled verifier rejected its build-time publisher" })
  yield* fs.writeFile(artifact, Buffer.alloc(bytes.length))
  if ((yield* verify) === 0) return yield* new AcceptanceFailed({ message: "The compiled verifier accepted changed installer bytes" })
  if (yield* fs.exists(profile)) return yield* new AcceptanceFailed({ message: "Finite verification created application state" })
  yield* Effect.logInfo("Compiled bootstrap verification accepted signed bytes, rejected tampering and left application state untouched")
}))
BunRuntime.runMain(run.pipe(Effect.provide(BunContext.layer)))
