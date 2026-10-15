/**
 * Forms every native kernel the engine ships with each GPU backend's real toolchain, and every
 * native kernel every supported catalog model requests.
 *
 *   bun run kernels:form [vulkan|cuda|metal]
 *
 * Without a backend, forms each one this host can: Vulkan everywhere, CUDA where the release stages
 * NVRTC (Linux and Windows), Metal on macOS. A named backend this host cannot form fails. Each
 * backend keeps a ledger of what its runs established under `target/kernel-formation/`, so a run
 * forms only what changed since; CI restores and saves it.
 */
import * as Command from "@effect/platform/Command"
import * as FileSystem from "@effect/platform/FileSystem"
import { BunContext } from "@effect/platform-bun"
import { currentHost, hostById } from "@magnitudedev/release/targets"
import { Effect, Option, Schema } from "effect"
import { dirname, resolve } from "node:path"
import { stageNvrtc } from "../../packages/release/scripts/build/nvrtc"
import { INFERENCE_ROOT } from "./compile"

class KernelFormationFailed extends Schema.TaggedError<KernelFormationFailed>()("KernelFormationFailed", {
  message: Schema.String,
}) {}

const Backend = Schema.Literal("vulkan", "cuda", "metal")
type Backend = typeof Backend.Type

const CATALOG = resolve(INFERENCE_ROOT, "target/catalog-inputs/model-planner-inputs.bundle")
const LEDGERS = resolve(INFERENCE_ROOT, "target/kernel-formation")

const formBackend = (backend: Backend, environment: Readonly<Record<string, string>>) =>
  Command.make(
    "cargo",
    "run", "--release", "--locked", "-p", "magnitude-kernel-formation", "--",
    backend,
    "--ledger", resolve(LEDGERS, `${backend}.ledger`),
    "--catalog", CATALOG,
  ).pipe(
    Command.workingDirectory(INFERENCE_ROOT),
    Command.env(environment),
    Command.stdout("inherit"),
    Command.stderr("inherit"),
    Command.exitCode,
    Effect.mapError((cause) => new KernelFormationFailed({ message: `kernel formation: ${String(cause)}` })),
    Effect.filterOrFail(
      (code) => code === 0,
      (code) => new KernelFormationFailed({ message: `${backend} kernel formation exited ${code}` }),
    ),
  )

const program = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const requested = process.argv[2]
  const host = hostById(currentHost())
  const backends: readonly Backend[] = requested === undefined
    ? [
        "vulkan",
        ...(Option.isSome(host.nvrtc) ? ["cuda" as const] : []),
        ...(process.platform === "darwin" ? ["metal" as const] : []),
      ]
    : [yield* Schema.decodeUnknown(Backend)(requested).pipe(
      Effect.mapError(() => new KernelFormationFailed({ message: `unknown backend ${requested}; expected vulkan, cuda or metal` })),
    )]
  if (!(yield* fs.exists(CATALOG).pipe(Effect.orElseSucceed(() => false)))) {
    return yield* new KernelFormationFailed({
      message: `no release header bundle at ${CATALOG}; build it with \`bun run icn:catalog:build-bundle\``,
    })
  }
  const nvrtc = yield* Option.match(host.nvrtc, {
    onNone: () => Effect.succeed<Readonly<Record<string, string>>>({}),
    onSome: (redistributable) => stageNvrtc(redistributable, resolve(INFERENCE_ROOT, "target/nvrtc")).pipe(
      Effect.map((staged) => ({ SEISMIC_NVRTC_DIRECTORY: dirname(staged.libraries[0]!) })),
      Effect.mapError((cause) => new KernelFormationFailed({ message: cause.message })),
    ),
  })
  for (const backend of backends) {
    yield* formBackend(backend, nvrtc)
  }
})

Effect.runPromise(program.pipe(Effect.provide(BunContext.layer))).catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`)
  process.exit(1)
})
