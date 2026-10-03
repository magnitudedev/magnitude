/**
 * Forms every native kernel the engine ships with each GPU backend's real toolchain, and every
 * native kernel every supported catalog model requests.
 *
 *   bun run kernels:form [vulkan|cuda|metal]
 *
 * Runs what this host can form: Vulkan everywhere, CUDA where the release stages NVRTC (Linux and
 * Windows), Metal on macOS. `MAGNITUDE_REQUIRE_KERNEL_FORMATION=1` makes a toolchain or catalog
 * header bundle this host lacks a failure instead of a skip.
 */
import * as Command from "@effect/platform/Command"
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

/** Each test binary and the test-name prefix its backend filter selects. */
const SUITES = [
  { package: "magnitude-kernels", target: ["--test", "formation"], suffix: "_kernels_form" },
  { package: "magnitude-service-server", target: ["--lib"], suffix: "_catalog_kernels_form" },
] as const

const cargoTest = (arguments_: readonly string[], environment: Readonly<Record<string, string>>) =>
  Command.make("cargo", ...arguments_).pipe(
    Command.workingDirectory(INFERENCE_ROOT),
    Command.env(environment),
    Command.stdout("inherit"),
    Command.stderr("inherit"),
    Command.exitCode,
    Effect.mapError((cause) => new KernelFormationFailed({ message: `cargo test: ${String(cause)}` })),
    Effect.filterOrFail(
      (code) => code === 0,
      (code) => new KernelFormationFailed({ message: `cargo ${arguments_.join(" ")} exited ${code}` }),
    ),
  )

const program = Effect.gen(function* () {
  const requested = process.argv[2]
  const backend = requested === undefined
    ? Option.none()
    : Option.some(yield* Schema.decodeUnknown(Backend)(requested).pipe(
      Effect.mapError(() => new KernelFormationFailed({ message: `unknown backend ${requested}; expected vulkan, cuda or metal` })),
    ))
  const nvrtc = yield* Option.match(hostById(currentHost()).nvrtc, {
    onNone: () => Effect.succeed<Readonly<Record<string, string>>>({}),
    onSome: (redistributable) => stageNvrtc(redistributable, resolve(INFERENCE_ROOT, "target/nvrtc")).pipe(
      Effect.map((staged) => ({ SEISMIC_NVRTC_DIRECTORY: dirname(staged.libraries[0]!) })),
      Effect.mapError((cause) => new KernelFormationFailed({ message: cause.message })),
    ),
  })
  for (const suite of SUITES) {
    const filter = Option.match(backend, {
      onNone: () => [suite.suffix],
      onSome: (name) => [`${name}${suite.suffix}`],
    })
    yield* cargoTest(
      ["test", "--release", "--locked", "-p", suite.package, ...suite.target, "--", ...filter, "--nocapture"],
      nvrtc,
    )
  }
})

Effect.runPromise(program.pipe(Effect.provide(BunContext.layer))).catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`)
  process.exit(1)
})
