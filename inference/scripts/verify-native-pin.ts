import { Data, Effect, Option, Schema } from "effect";
import { resolve } from "node:path";

class NativePinMismatch extends Data.TaggedError("NativePinMismatch")<{
  readonly message: string;
}> {}

const NativePin = Schema.Struct({
  llama_cpp_rs: Schema.Struct({
    repository: Schema.String,
    revision: Schema.String,
    release: Schema.String,
  }),
  llama_cpp: Schema.Struct({ repository: Schema.String, revision: Schema.String }),
});

const CargoManifest = Schema.Struct({
  workspace: Schema.Struct({
    dependencies: Schema.Struct({
      "llama-cpp-2": Schema.Struct({
        git: Schema.String,
        rev: Schema.String,
      }),
    }),
  }),
});

const CargoLock = Schema.Struct({
  package: Schema.Array(
    Schema.Struct({
      name: Schema.String,
      version: Schema.String,
      source: Schema.optionalWith(Schema.String, { exact: true, as: "Option" }),
    })
  ),
});

const root = resolve(import.meta.dirname, "..");
const readToml = (path: string) =>
  Effect.tryPromise({
    try: async () => Bun.TOML.parse(await Bun.file(path).text()),
    catch: (cause) =>
      new NativePinMismatch({
        message: cause instanceof Error ? cause.message : String(cause),
      }),
  });

const fail = (message: string) => new NativePinMismatch({ message });

const program = Effect.gen(function* () {
  const pin = yield* readToml(resolve(root, "native-pin.toml")).pipe(
    Effect.flatMap(Schema.decodeUnknown(NativePin)),
    Effect.mapError((cause) => fail(String(cause)))
  );
  const manifest = yield* readToml(resolve(root, "Cargo.toml")).pipe(
    Effect.flatMap(Schema.decodeUnknown(CargoManifest)),
    Effect.mapError((cause) => fail(String(cause)))
  );
  const lock = yield* readToml(resolve(root, "Cargo.lock")).pipe(
    Effect.flatMap(Schema.decodeUnknown(CargoLock)),
    Effect.mapError((cause) => fail(String(cause)))
  );
  const dependency = manifest.workspace.dependencies["llama-cpp-2"];
  if (dependency.git !== pin.llama_cpp_rs.repository)
    return yield* fail("Cargo repository does not match native-pin.toml");
  if (dependency.rev !== pin.llama_cpp_rs.revision)
    return yield* fail("Cargo revision does not match native-pin.toml");

  const locked = Option.fromNullable(
    lock.package.find((entry) => entry.name === "llama-cpp-2")
  );
  if (Option.isNone(locked)) return yield* fail("llama-cpp-2 is absent from Cargo.lock");
  if (locked.value.version !== pin.llama_cpp_rs.release)
    return yield* fail("Cargo.lock binding version does not match the recorded release");
  const expectedSource = `git+${pin.llama_cpp_rs.repository}?rev=${pin.llama_cpp_rs.revision}#${pin.llama_cpp_rs.revision}`;
  if (!Option.contains(locked.value.source, expectedSource))
    return yield* fail("Cargo.lock does not resolve the exact recorded binding revision");
  if (!/^[0-9a-f]{40}$/.test(pin.llama_cpp.revision))
    return yield* fail("the embedded llama.cpp revision is not a full commit SHA");

  console.log(
    `llama-cpp-rs ${pin.llama_cpp_rs.release} ${pin.llama_cpp_rs.revision.slice(0, 12)} -> llama.cpp ${pin.llama_cpp.revision.slice(0, 12)}`
  );
});

await Effect.runPromise(program);
