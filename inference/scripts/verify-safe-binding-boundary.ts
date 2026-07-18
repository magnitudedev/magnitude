import { Data, Effect, Schema } from "effect";
import { relative, resolve } from "node:path";

class SafeBindingBoundaryError extends Data.TaggedError(
  "SafeBindingBoundaryError"
)<{
  readonly message: string;
}> {}

const CargoDependency = Schema.Struct({
  name: Schema.String,
});

const CargoPackage = Schema.Struct({
  id: Schema.String,
  name: Schema.String,
  manifest_path: Schema.String,
  dependencies: Schema.Array(CargoDependency),
});

const CargoMetadata = Schema.Struct({
  packages: Schema.Array(CargoPackage),
  workspace_members: Schema.Array(Schema.String),
});

const inferenceRoot = resolve(import.meta.dirname, "..");
const bindingsRoot = resolve(inferenceRoot, "native/llama-cpp-rs");
const safeCrateSource = resolve(bindingsRoot, "llama-cpp-2/src");

const fail = (message: string) => new SafeBindingBoundaryError({ message });

const read = (path: string) =>
  Effect.tryPromise({
    try: () => Bun.file(path).text(),
    catch: (cause) => fail(`Unable to read ${path}: ${String(cause)}`),
  });

const cargoMetadata = Effect.tryPromise({
  try: async () => {
    const process = Bun.spawn(
      [
        "cargo",
        "metadata",
        "--manifest-path",
        resolve(inferenceRoot, "Cargo.toml"),
        "--format-version",
        "1",
        "--no-deps",
      ],
      { stdout: "pipe", stderr: "pipe" }
    );
    const [stdout, stderr, exitCode] = await Promise.all([
      new Response(process.stdout).text(),
      new Response(process.stderr).text(),
      process.exited,
    ]);
    if (exitCode !== 0) {
      throw new Error(`cargo metadata exited ${exitCode}: ${stderr.trim()}`);
    }
    return stdout;
  },
  catch: (cause) => fail(`Unable to inspect Cargo metadata: ${String(cause)}`),
}).pipe(
  Effect.flatMap(Schema.decode(Schema.parseJson(CargoMetadata))),
  Effect.mapError((cause) => fail(`Invalid Cargo metadata: ${String(cause)}`))
);

const program = Effect.gen(function* () {
  const metadata = yield* cargoMetadata;
  const workspaceMembers = new Set(metadata.workspace_members);
  const violations = metadata.packages
    .filter((entry) => workspaceMembers.has(entry.id))
    .filter((entry) =>
      entry.dependencies.some((dependency) => dependency.name === "llama-cpp-sys-2")
    )
    .map(
      (entry) =>
        `${entry.name} (${relative(inferenceRoot, entry.manifest_path)}) directly depends on llama-cpp-sys-2`
    );

  const sourcePaths = yield* Effect.tryPromise({
    try: async () => {
      const paths: Array<string> = [];
      for await (const path of new Bun.Glob("**/*.rs").scan({
        cwd: safeCrateSource,
        onlyFiles: true,
      })) {
        paths.push(path);
      }
      return paths.sort();
    },
    catch: (cause) =>
      fail(`Unable to enumerate ${safeCrateSource}: ${String(cause)}`),
  });

  for (const sourcePath of sourcePaths) {
    const absolutePath = resolve(safeCrateSource, sourcePath);
    const source = yield* read(absolutePath);
    if (/pub\s+(?:use|extern\s+crate)\s+llama_cpp_sys_2\b/.test(source)) {
      violations.push(`${sourcePath} publicly re-exports llama-cpp-sys-2`);
    }

    const rawAccessor = /pub\s+(?:unsafe\s+)?fn\s+(as_ptr|as_raw|into_raw|from_raw)\b/g;
    for (const match of source.matchAll(rawAccessor)) {
      const offset = match.index ?? 0;
      const signature = source.slice(offset, offset + 500).split("{")[0] ?? "";
      if (
        /llama_cpp_sys_2::|\*\s*(?:const|mut)\s+(?:llama_|mtmd_)|NonNull\s*<\s*(?:llama_cpp_sys_2::)?(?:llama_|mtmd_)/.test(
          signature
        )
      ) {
        violations.push(
          `${sourcePath} exposes native owner through ${match[1] ?? "raw accessor"}`
        );
      }
    }
  }

  if (violations.length > 0) {
    return yield* fail(
      `Safe bindings boundary violations:\n- ${violations.join("\n- ")}`
    );
  }

  console.log(
    `Verified ${metadata.workspace_members.length} inference workspace packages consume only the safe llama-cpp-2 boundary`
  );
});

Effect.runPromise(program).catch((error) => {
  console.error(error);
  process.exit(1);
});
