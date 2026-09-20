import { FileSystem } from "@effect/platform"
import { Effect, Redacted, Schema } from "effect"
import { Digest, InfrastructureFailure } from "../domain"
import { checkedCommand } from "../process"
import { sha256 } from "../snapshot"
import { InitializationDownload, renderLinuxInitialization, LinuxInitialization } from "./linux-initialization"

const PinnedFile = Schema.Struct({ file: Schema.NonEmptyString, sha256: Digest })
export const LinuxAzureInitialization = Schema.Struct({ kind: Schema.Literal("linux"),
  setup: PinnedFile,
  distribution: LinuxInitialization.fields.distribution,
  adminUsername: LinuxInitialization.fields.adminUsername,
  architecture: LinuxInitialization.fields.architecture,
  node: InitializationDownload, rustup: InitializationDownload,
  runtime: Schema.Struct({ account: Schema.String.pipe(Schema.pattern(/^[a-z0-9]{3,24}$/)),
    container: Schema.String.pipe(Schema.pattern(/^[a-z0-9](?:[a-z0-9-]{1,61})[a-z0-9]$/)),
    blob: Schema.String.pipe(Schema.pattern(/^worker-runtime\/[a-f0-9]{64}\.tar\.gz$/)),
    sha256: Digest, bytes: Schema.Int.pipe(Schema.between(1, 1024 ** 3)) }),
})
export const AzureInitialization = Schema.Union(PinnedFile, LinuxAzureInitialization)
export type AzureInitialization = typeof AzureInitialization.Type
const fail = (message: string) => new InfrastructureFailure({ operation: "azure-initialization", message })

/** Identity pins the recipe and bytes, independently of the short-lived read capability. */
export const prepareAzureInitialization = (initialization: AzureInitialization, scope: {
  readonly executable: string; readonly subscription: string; readonly adminUsername: string; readonly architecture: "x64" | "arm64"; readonly os: string; readonly version: string
}) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const pinned = "kind" in initialization ? initialization.setup : initialization
  if (Number((yield* fs.stat(pinned.file)).size) > 64 * 1024) return yield* fail("Worker initialization exceeds Azure's 64 KiB limit")
  const bytes = yield* fs.readFile(pinned.file)
  if (bytes.byteLength > 64 * 1024 || sha256(bytes) !== pinned.sha256) return yield* fail("Worker initialization digest differs from configured runtime")
  if (!("kind" in initialization)) return { customData: Buffer.from(bytes).toString("base64"), identity: pinned.sha256 }
  const recipe = initialization
  if (recipe.distribution.os !== scope.os || recipe.distribution.version !== scope.version) return yield* fail("Linux initialization distribution differs from allocation")
  if (recipe.adminUsername !== scope.adminUsername || recipe.architecture !== scope.architecture) return yield* fail("Linux initialization user or architecture differs from allocation")
  if (recipe.runtime.blob !== `worker-runtime/${recipe.runtime.sha256}.tar.gz`) return yield* fail("Runtime blob name differs from its pinned digest")
  const identity = sha256(yield* Schema.encode(Schema.parseJson(LinuxAzureInitialization))(recipe))
  // Cloud-init has a twenty-minute limit. A one-hour blob-only read grant covers startup;
  // neither storage account keys nor the coordinator identity enter the guest.
  const now = Math.floor(Date.now() / 1000) * 1000
  const timestamp = (milliseconds: number) => new Date(milliseconds).toISOString().replace(".000Z", "Z")
  const expiry = timestamp(now + 60 * 60_000)
  const signed = yield* checkedCommand(scope.executable, ["storage", "blob", "generate-sas", "--subscription", scope.subscription,
    "--account-name", recipe.runtime.account, "--container-name", recipe.runtime.container, "--name", recipe.runtime.blob,
    "--as-user", "--auth-mode", "login", "--permissions", "r", "--https-only", "--full-uri",
    "--start", timestamp(now - 5 * 60_000), "--expiry", expiry, "--only-show-errors", "--output", "tsv"],
  { timeoutMs: 30_000, maxOutputBytes: 16 * 1024 }).pipe(Effect.mapError(() => fail("Cannot issue the worker runtime download capability")))
  const url = yield* Effect.try({ try: () => new URL(signed.stdout.trim()), catch: () => fail("Malformed runtime download capability") })
  if (url.protocol !== "https:" || url.host !== `${recipe.runtime.account}.blob.core.windows.net` || url.username || url.password || url.hash ||
    url.pathname !== `/${recipe.runtime.container}/${recipe.runtime.blob}` || url.searchParams.get("sp") !== "r" ||
    url.searchParams.get("sr") !== "b" || url.searchParams.get("spr") !== "https" || !url.searchParams.get("sig") ||
    !url.searchParams.get("skoid") || !url.searchParams.get("sktid") ||
    Date.parse(url.searchParams.get("se") ?? "") !== Date.parse(expiry)) return yield* fail("Runtime download capability has unexpected scope or expiry")
  const cloudInit = yield* renderLinuxInitialization({ distribution: recipe.distribution, adminUsername: recipe.adminUsername, architecture: recipe.architecture,
    node: recipe.node, rustup: recipe.rustup,
    runtime: { url: Redacted.make(url.toString()), sha256: recipe.runtime.sha256, bytes: recipe.runtime.bytes },
  }, new TextDecoder().decode(bytes))
  return { customData: Buffer.from(cloudInit).toString("base64"), identity }
}).pipe(Effect.mapError(error => error._tag === "InfrastructureFailure" ? error : fail("Cannot prepare configured worker initialization")))
