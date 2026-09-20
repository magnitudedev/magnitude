import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Schema } from "effect"
import { expect, test } from "vitest"
import pins from "../tools/nvidia-drivers.json"
import { NvidiaPreparation, nvidiaPreparationScript, nvidiaReadinessScript } from "../src/providers/nvidia-preparation"
import { checkedCommand, ProcessExecutorLive } from "../src/process"

for (const [name, raw] of Object.entries(pins)) test(`${name} recipe is pinned and generated Linux code parses`, () => Effect.runPromise(Effect.gen(function* () {
  const pin = yield* Schema.decodeUnknown(NvidiaPreparation)(raw)
  const script = yield* nvidiaPreparationScript(pin, name.startsWith("linux") ? "Linux" : "Windows")
  if (name.startsWith("linux")) {
    yield* checkedCommand("/bin/bash", ["-n", "-c", script])
    const python = script.split("<<'LAB_GPU'\n")[1]!.split("\nLAB_GPU")[0]!
    yield* checkedCommand("python3", ["-c", "import ast,sys;ast.parse(sys.argv[1])", python])
  }
}).pipe(Effect.provide(ProcessExecutorLive))))

for (const [output, exit, passes] of [
  ["NVIDIA A10-24Q, 570.211.01", 0, true],
  ["NVIDIA A10-24Q, 535.161.08", 0, false],
  ["NVIDIA RTX PRO 6000, 570.211.01", 0, false],
  ["NVIDIA A100, 570.211.01", 0, false],
  ["NVIDIA A10, 570.211.01\nNVIDIA A10, 570.211.01", 0, false],
  ["NVIDIA A10, 570.211.01", 1, false],
  ["", 0, false],
] as const) test(`native readiness rejects mismatched or failed observations: ${output}/${exit}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const directory = yield* fs.makeTempDirectoryScoped({ prefix: "lab-gpu-ready-" })
  yield* fs.writeFileString(`${directory}/nvidia-smi`, `#!/bin/sh\nprintf '%s\\n' '${output}'\nexit ${exit}\n`, { mode: 0o755 })
  const pin = yield* Schema.decodeUnknown(NvidiaPreparation)(pins["linux-a10"])
  const result = yield* checkedCommand("/bin/sh", ["-c", nvidiaReadinessScript(pin, "Linux")], {
    env: { PATH: `${directory}:/usr/bin:/bin:/usr/local/bin` },
  }).pipe(Effect.either)
  expect(result._tag).toBe(passes ? "Right" : "Left")
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))))

test("GPU configuration rejects private capabilities and script injection", () => {
  for (const change of [{ version: "570;evil" }, { download: { ...pins["linux-a10"].download, url: "https://download.microsoft.com/download/driver?sig=secret" } }])
    expect(() => Schema.decodeUnknownSync(NvidiaPreparation)({ ...pins["linux-a10"], ...change })).toThrow()
})
