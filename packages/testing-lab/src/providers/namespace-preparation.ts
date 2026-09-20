import { FileSystem } from "@effect/platform"
import { DateTime, Effect, Option, Schema } from "effect"
import { packageManager } from "../../../../package.json"
import { Digest, InfrastructureFailure } from "../domain"
import { type NamespaceMachine } from "../machines"
import { checkedCommand } from "../process"
import { sha256 } from "../snapshot"
import { retainWorkerDiagnostic } from "../worker-diagnostics"
import { AzureRuntimeBlob, azureRuntimeDownload } from "./azure-initialization"
import { InitializationDownload } from "./linux-initialization"
import { type NamespaceImage } from "./namespace"

export const NamespacePreparation = Schema.Struct({
  setup: Schema.Struct({ file: Schema.NonEmptyString, sha256: Digest }),
  runtime: AzureRuntimeBlob, node: InitializationDownload, rustup: InitializationDownload, tirith: InitializationDownload,
  azureExecutable: Schema.NonEmptyString, subscription: Schema.NonEmptyString,
})
export type NamespacePreparation = typeof NamespacePreparation.Type
const fail = (message: string) => new InfrastructureFailure({ operation: "namespace-preparation", message })

/** Only the trusted runtime's read capability crosses this boundary, never provider credentials. */
export const prepareNamespaceMachine = (executable: string, recipe: NamespacePreparation, machine: typeof NamespaceMachine.Type, image: NamespaceImage, workKind: "build" | "test") => Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const bytes = yield* fs.readFile(recipe.setup.file)
  if (bytes.byteLength > 64 * 1024 || sha256(bytes) !== recipe.setup.sha256) return yield* fail("Mac preparation script differs from its pin")
  const identity = sha256(yield* Schema.encode(Schema.parseJson(Schema.Struct({ recipe: NamespacePreparation,
    productVersion: Schema.String, buildVersion: Schema.String, workKind: Schema.Literal("build", "test") })) )({ recipe, workKind, productVersion: image.productVersion, buildVersion: image.buildVersion }))
  const remaining = DateTime.toEpochMillis(machine.tags.expiresAt) - Date.now()
  if (remaining <= 0) return yield* fail("Mac preparation lease has expired")
  const directory = yield* fs.makeTempDirectoryScoped({ prefix: "lab-mac-preparation-" })
  const remote = `/Users/runner/.magnitude-lab-prepare/${machine.tags.leaseId}`
  const runtime = yield* azureRuntimeDownload(recipe.runtime, { executable: recipe.azureExecutable, subscription: recipe.subscription })
  const config = yield* Schema.encode(Schema.parseJson(Schema.Struct({ identity: Digest, productVersion: Schema.String, buildVersion: Schema.String,
    workKind: Schema.Literal("build", "test"), bunVersion: Schema.String, runtime: InitializationDownload, node: InitializationDownload, rustup: InitializationDownload, tirith: InitializationDownload })))({
    identity, workKind, productVersion: image.productVersion, buildVersion: image.buildVersion, bunVersion: packageManager.replace(/^bun@/, ""),
    runtime, node: recipe.node, rustup: recipe.rustup, tirith: recipe.tirith,
  })
  yield* fs.writeFile(`${directory}/setup.sh`, bytes, { mode: 0o600 })
  yield* fs.writeFileString(`${directory}/config.json`, config, { mode: 0o600 })
  const execute = (args: readonly string[], timeoutMs = 120_000) => checkedCommand(executable, ["exec", machine.name, "--", ...args], { timeoutMs })
  const prepare = Effect.gen(function* () {
    yield* execute(["/bin/mkdir", "-p", remote])
    yield* execute(["/bin/chmod", "700", remote])
    for (const name of ["setup.sh", "config.json"]) yield* checkedCommand(executable, ["upload", machine.name, `${directory}/${name}`, `${remote}/${name}`], { timeoutMs: 120_000 })
    yield* execute(["/bin/bash", "-c", 'umask 077; exec /bin/bash "$1/setup.sh" "$1/config.json" >> "$1/output.log" 2>&1', "lab-prepare", remote], Math.min(30 * 60_000, remaining))
    // A fresh command proves current desktop/user identity as well as the immutable receipt.
    yield* execute(["/opt/homebrew/bin/python3", "-c", `import json,pathlib,os,subprocess; p=pathlib.Path('/var/db/magnitude-lab/runtime.json'); s=p.stat(); r=json.loads(p.read_text()); assert s.st_uid==0 and not s.st_mode&0o022; assert r['identity']=='${identity}' and r['uid']==os.getuid(); assert subprocess.check_output(['stat','-f','%Su','/dev/console'],text=True).strip()=='runner'; assert pathlib.Path('/Users/runner/lab-runtime/worker').is_file(); print('Mac runtime ready')`])
  })
  yield* prepare.pipe(Effect.timeoutFail({ duration: Math.max(1, Math.min(30 * 60_000, DateTime.toEpochMillis(machine.tags.expiresAt) - Date.now())),
    onTimeout: () => fail("Mac preparation exceeded its lease deadline") }), Effect.catchAll(() => Effect.gen(function* () {
    const diagnostic = yield* execute(["/usr/bin/tail", "-c", "24000", `${remote}/output.log`]).pipe(
      Effect.flatMap(output => retainWorkerDiagnostic(machine, "initialization", output.stdout + "\n" + output.stderr)), Effect.either)
    return yield* new InfrastructureFailure({ operation: "namespace-preparation", message: diagnostic._tag === "Right"
      ? "Mac preparation failed; initialization diagnostics retained" : "Mac preparation failed; diagnostics unavailable",
      evidence: diagnostic._tag === "Right" ? Option.some([diagnostic.right]) : Option.none() })
  })), Effect.ensuring(execute(["/bin/rm", "-f", `${remote}/config.json`]).pipe(Effect.orDie)))
})).pipe(Effect.mapError(error => error._tag === "InfrastructureFailure" ? error : fail("Cannot prepare Mac runtime")))
