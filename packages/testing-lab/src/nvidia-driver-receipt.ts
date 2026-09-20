import { FileSystem } from "@effect/platform"
import { Effect, Option, Schema, Stream } from "effect"
import { createHash } from "node:crypto"
import pins from "../tools/nvidia-drivers.json"
import { AssertionFailure, Digest } from "./domain"

export const NvidiaDriverReceipt = Schema.Struct({ schemaVersion: Schema.Literal(1),
  model: Schema.Literal("a10", "rtx-pro-6000"), version: Schema.String, installerSha256: Digest,
  path: Schema.String, sha256: Digest, bytes: Schema.Int.pipe(Schema.between(1, 2 * 1024 ** 3)),
})
export const NvidiaDriverProvenance = Schema.Struct({ kind: Schema.Literal("nvidia-driver"),
  model: NvidiaDriverReceipt.fields.model, version: Schema.String, installerSha256: Digest, sha256: Digest,
})
export const nvidiaReceiptPath = "/var/lib/magnitude-lab-driver/receipt.json"
const fail = (message: string) => new AssertionFailure({ message: `NVIDIA driver provenance: ${message}` })

/** Only the administrator's pinned installer can establish an unpackaged driver boundary. */
export const verifyNvidiaDriverReceipt = (path: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  for (const entry of ["/var", "/var/lib", "/var/lib/magnitude-lab-driver", nvidiaReceiptPath]) {
    const stat = yield* fs.stat(entry)
    if (Option.getOrUndefined(stat.uid) !== 0 || (stat.mode & 0o022) !== 0
      || stat.type !== (entry === nvidiaReceiptPath ? "File" : "Directory")
      || (yield* fs.realPath(entry)) !== entry) return yield* fail("receipt ownership or path is unsafe")
    if (entry === nvidiaReceiptPath && stat.size > 16 * 1024) return yield* fail("receipt exceeds size limit")
  }
  const receipt = yield* fs.readFileString(nvidiaReceiptPath).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(NvidiaDriverReceipt))),
    Effect.mapError(() => fail("missing or invalid installation receipt")))
  const pin = pins[`linux-${receipt.model}`]
  if (receipt.version !== pin.version || receipt.installerSha256 !== pin.download.sha256 || receipt.path !== path
    || !/^\/(?:usr\/)?lib(?:64)?\//.test(path)) return yield* fail("installed driver differs from its pinned provenance")
  const stat = yield* fs.stat(path)
  if (stat.type !== "File" || Option.getOrUndefined(stat.uid) !== 0 || (stat.mode & 0o022) !== 0
    || Number(stat.size) !== receipt.bytes || (yield* fs.realPath(path)) !== path) return yield* fail("driver file differs from its installation receipt")
  const digest = createHash("sha256")
  yield* fs.stream(path).pipe(Stream.runForEach(bytes => Effect.sync(() => { digest.update(bytes) })))
  if (digest.digest("hex") !== receipt.sha256) return yield* fail("installed driver bytes changed")
  return NvidiaDriverProvenance.make({ kind: "nvidia-driver", model: receipt.model, version: receipt.version,
    installerSha256: receipt.installerSha256, sha256: receipt.sha256 })
})
