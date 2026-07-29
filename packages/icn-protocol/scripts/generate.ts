import * as FileSystem from "@effect/platform/FileSystem";
import * as BunContext from "@effect/platform-bun/BunContext";
import * as BunRuntime from "@effect/platform-bun/BunRuntime";
import { compileOpenApi } from "@magnitudedev/openapi-effect";
import { Data, Effect, HashMap, Option } from "effect";
import { resolve } from "node:path";
import { config } from "../openapi-effect.config.js";

class OpenApiExportError extends Data.TaggedError("OpenApiExportError")<{
  readonly message: string;
}> {}

class GeneratedOutputMismatch extends Data.TaggedError(
  "GeneratedOutputMismatch"
)<{ readonly files: ReadonlyArray<string> }> {}

const packageRoot = resolve(import.meta.dirname, "..");
const repositoryRoot = resolve(packageRoot, "../..");
const generatedRoot = resolve(packageRoot, "src/generated");

const exportOpenApi = Effect.tryPromise({
  try: async () => {
    const process = Bun.spawn(
      [
        "cargo",
        "run",
        "--quiet",
        "--manifest-path",
        resolve(repositoryRoot, "inference/Cargo.toml"),
        "-p",
        "icn-api",
        "--bin",
        "export-openapi",
      ],
      { cwd: repositoryRoot, stdout: "pipe", stderr: "pipe" }
    );
    const [status, stdout, stderr] = await Promise.all([
      process.exited,
      new Response(process.stdout).text(),
      new Response(process.stderr).text(),
    ]);
    if (status !== 0) throw new Error(stderr || `cargo exited with ${status}`);
    return { document: JSON.parse(stdout), source: `${stdout.trimEnd()}\n` };
  },
  catch: (cause) =>
    new OpenApiExportError({
      message: cause instanceof Error ? cause.message : String(cause),
    }),
});

const isCheck = Option.fromNullable(process.argv[2]).pipe(
  Option.contains("--check")
);

const program = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem;
  const exported = yield* exportOpenApi;
  const project = yield* compileOpenApi(exported.document, config);
  const files = [...HashMap.entries(project.files)].sort(([left], [right]) =>
    left.localeCompare(right)
  );

  if (isCheck) {
    const mismatches: Array<string> = [];
    const openApi = yield* fs
      .readFileString(resolve(packageRoot, "openapi.json"))
      .pipe(Effect.option);
    if (Option.isNone(openApi) || openApi.value !== exported.source)
      mismatches.push("openapi.json");
    for (const [path, source] of files) {
      const existing = yield* fs
        .readFileString(resolve(generatedRoot, path))
        .pipe(Effect.option);
      if (Option.isNone(existing) || existing.value !== source)
        mismatches.push(path);
    }
    if (mismatches.length > 0)
      return yield* new GeneratedOutputMismatch({ files: mismatches });
    return;
  }

  yield* fs.makeDirectory(generatedRoot, { recursive: true });
  yield* fs.writeFileString(resolve(packageRoot, "openapi.json"), exported.source);
  for (const [path, source] of files)
    yield* fs.writeFileString(resolve(generatedRoot, path), source);
});

BunRuntime.runMain(program.pipe(Effect.provide(BunContext.layer)));
