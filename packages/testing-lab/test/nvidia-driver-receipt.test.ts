import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Option } from "effect"
import { createHash } from "node:crypto"
import { join } from "node:path"
import { expect, test } from "vitest"
import pins from "../tools/nvidia-drivers.json"
import { nvidiaReceiptPath, verifyNvidiaDriverReceipt } from "../src/nvidia-driver-receipt"

for (const mode of ["valid", "tampered", "installer", "version", "path", "owner", "writable", "symlink", "oversized"] as const) {
  test(`NVIDIA installer provenance: ${mode}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-driver-receipt-" })
    const file = join(root, "driver"), path = "/usr/lib/x86_64-linux-gnu/libcuda.so.570.211.01"
    const bytes = Buffer.from("pinned driver payload")
    yield* fs.writeFile(file, bytes)
    const stat = yield* fs.stat(file)
    const pin = pins["linux-a10"]
    const receipt = { schemaVersion: 1, model: "a10", version: mode === "version" ? "000.00" : pin.version,
      installerSha256: mode === "installer" ? "0".repeat(64) : pin.download.sha256,
      path: mode === "path" ? "/usr/lib/other" : path,
      sha256: createHash("sha256").update(mode === "tampered" ? "changed" : bytes).digest("hex"), bytes: bytes.length }
    const result = yield* verifyNvidiaDriverReceipt(path).pipe(Effect.provideService(FileSystem.FileSystem, { ...fs,
      stat: entry => Effect.succeed({ ...stat, uid: Option.some(mode === "owner" ? 1000 : 0),
        mode: mode === "writable" ? 0o666 : 0o444,
        size: FileSystem.Size(mode === "oversized" && entry === nvidiaReceiptPath ? 20000 : bytes.length),
        type: entry === path || entry === nvidiaReceiptPath ? "File" : "Directory" }),
      realPath: entry => Effect.succeed(mode === "symlink" ? "/other" : entry),
      readFileString: () => Effect.succeed(JSON.stringify(receipt)),
      stream: () => fs.stream(file),
    }), Effect.either)
    expect(result._tag).toBe(mode === "valid" ? "Right" : "Left")
    if (result._tag === "Right") expect(result.right.kind).toBe("nvidia-driver")
  })).pipe(Effect.provide(BunContext.layer))))
}
