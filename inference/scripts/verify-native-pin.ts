import { Data, Effect, Option, Schema } from "effect";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";

export class NativePinVerificationError extends Data.TaggedError(
  "NativePinVerificationError"
)<{
  readonly message: string;
}> {}

const CommitRevision = Schema.String.pipe(
  Schema.pattern(/^[0-9a-f]{40}$/)
);

const NativePinSchema = Schema.Struct({
  schema_version: Schema.Literal(1),
  llama_cpp_rs: Schema.Struct({
    checkout_path: Schema.String,
    crate_path: Schema.String,
    repository: Schema.String,
    upstream_repository: Schema.String,
    revision: CommitRevision,
    release: Schema.String,
  }),
  llama_cpp: Schema.Struct({
    checkout_path: Schema.String,
    repository: Schema.String,
    revision: CommitRevision,
  }),
});

export type NativePin = typeof NativePinSchema.Type;

const CargoManifestSchema = Schema.Struct({
  workspace: Schema.Struct({
    dependencies: Schema.Struct({
      "llama-cpp-2": Schema.Struct({
        path: Schema.String,
        git: Schema.optionalWith(Schema.String, { exact: true, as: "Option" }),
        rev: Schema.optionalWith(Schema.String, { exact: true, as: "Option" }),
      }),
    }),
  }),
});

const BindingManifestSchema = Schema.Struct({
  package: Schema.Struct({
    name: Schema.Literal("llama-cpp-2"),
    version: Schema.String,
    repository: Schema.String,
  }),
  dependencies: Schema.Struct({
    "llama-cpp-sys-2": Schema.Struct({
      path: Schema.String,
      version: Schema.String,
    }),
  }),
});

const SysManifestSchema = Schema.Struct({
  package: Schema.Struct({
    name: Schema.Literal("llama-cpp-sys-2"),
    version: Schema.String,
    repository: Schema.String,
  }),
});

const CargoLockSchema = Schema.Struct({
  package: Schema.Array(
    Schema.Struct({
      name: Schema.String,
      version: Schema.String,
      source: Schema.optionalWith(Schema.String, { exact: true, as: "Option" }),
    })
  ),
});

const inferenceRoot = resolve(import.meta.dirname, "..");
const repositoryRoot = resolve(inferenceRoot, "..");

const fail = (message: string) =>
  new NativePinVerificationError({ message });

const readText = (path: string) =>
  Effect.tryPromise({
    try: () => Bun.file(path).text(),
    catch: (cause) =>
      fail(
        `Unable to read ${path}: ${cause instanceof Error ? cause.message : String(cause)}`
      ),
  });

const readToml = (path: string) =>
  readText(path).pipe(
    Effect.map(Bun.TOML.parse),
    Effect.mapError((cause) => fail(`Unable to parse ${path}: ${cause.message}`))
  );

const decodeToml = <A, I>(schema: Schema.Schema<A, I>, path: string) =>
  readToml(path).pipe(
    Effect.flatMap(Schema.decodeUnknown(schema)),
    Effect.mapError((cause) => fail(`Invalid ${path}: ${String(cause)}`))
  );

const requirePathInside = (base: string, configuredPath: string) => {
  const absolutePath = resolve(base, configuredPath);
  const relativePath = relative(base, absolutePath);
  if (
    relativePath.length === 0 ||
    relativePath === ".." ||
    relativePath.startsWith(`..${sep}`) ||
    isAbsolute(relativePath)
  ) {
    return Effect.fail(fail(`Path escapes its root: ${configuredPath}`));
  }
  return Effect.succeed(absolutePath);
};

const requireExists = (path: string) =>
  Effect.promise(() => Bun.file(path).exists()).pipe(
    Effect.flatMap((exists) =>
      exists ? Effect.succeed(path) : Effect.fail(fail(`Missing ${path}`))
    )
  );

const readOptionalText = (path: string) =>
  Effect.promise(() => Bun.file(path).exists()).pipe(
    Effect.flatMap((exists) =>
      exists
        ? readText(path).pipe(Effect.map(Option.some))
        : Effect.succeed(Option.none<string>())
    )
  );

const resolveGitDirectory = (checkoutPath: string) =>
  Effect.gen(function* () {
    const dotGitPath = resolve(checkoutPath, ".git");
    const dotGit = yield* readText(dotGitPath);
    const match = Option.fromNullable(/^gitdir:\s*(.+)$/m.exec(dotGit));
    if (Option.isNone(match)) {
      return yield* fail(`${dotGitPath} is not a submodule gitdir file`);
    }
    const configuredGitDirectory = Option.fromNullable(match.value[1]);
    if (Option.isNone(configuredGitDirectory)) {
      return yield* fail(`${dotGitPath} has no gitdir value`);
    }
    const gitDirectory = resolve(dirname(dotGitPath), configuredGitDirectory.value.trim());
    yield* requireExists(resolve(gitDirectory, "HEAD"));
    return gitDirectory;
  });

const readPackedRef = (gitDirectory: string, refName: string) =>
  readOptionalText(resolve(gitDirectory, "packed-refs")).pipe(
    Effect.flatMap(
      Option.match({
        onNone: () =>
          Effect.fail(fail(`Git ref ${refName} is neither loose nor packed`)),
        onSome: (contents) => {
          const entry = Option.fromNullable(
            contents
              .split("\n")
              .map((line) => line.trim())
              .find((line) => line.endsWith(` ${refName}`))
          );
          if (Option.isNone(entry)) {
            return Effect.fail(fail(`Git ref ${refName} is absent from packed-refs`));
          }
          return Effect.succeed(entry.value.slice(0, 40));
        },
      })
    )
  );

const readCheckoutRevision = (checkoutPath: string) =>
  Effect.gen(function* () {
    const gitDirectory = yield* resolveGitDirectory(checkoutPath);
    const head = (yield* readText(resolve(gitDirectory, "HEAD"))).trim();
    if (/^[0-9a-f]{40}$/.test(head)) return head;
    if (!head.startsWith("ref: ")) {
      return yield* fail(`Unsupported HEAD value in ${gitDirectory}: ${head}`);
    }

    const refName = head.slice("ref: ".length);
    const looseRef = yield* readOptionalText(resolve(gitDirectory, refName));
    return yield* Option.match(looseRef, {
      onNone: () => readPackedRef(gitDirectory, refName),
      onSome: (revision) => Effect.succeed(revision.trim()),
    });
  });

const parseSubmoduleProperty = (
  contents: string,
  submoduleName: string,
  property: "path" | "url"
) => {
  const header = `[submodule "${submoduleName}"]`;
  const headerOffset = contents.indexOf(header);
  if (headerOffset < 0) return Option.none<string>();
  const bodyStart = headerOffset + header.length;
  const nextHeaderOffset = contents.indexOf("[submodule ", bodyStart);
  const body = contents.slice(
    bodyStart,
    nextHeaderOffset < 0 ? contents.length : nextHeaderOffset
  );
  const line = Option.fromNullable(
    body
      .split("\n")
      .map((candidate) => candidate.trim())
      .find((candidate) => candidate.startsWith(`${property} =`))
  );
  return Option.map(line, (value) => value.slice(value.indexOf("=") + 1).trim());
};

const verifySubmodule = (
  gitmodulesPath: string,
  submoduleName: string,
  expectedPath: string,
  expectedRepository: string
) =>
  Effect.gen(function* () {
    const contents = yield* readText(gitmodulesPath);
    const actualPath = parseSubmoduleProperty(contents, submoduleName, "path");
    const actualRepository = parseSubmoduleProperty(contents, submoduleName, "url");
    if (!Option.contains(actualPath, expectedPath)) {
      return yield* fail(
        `${gitmodulesPath} does not map ${submoduleName} to ${expectedPath}`
      );
    }
    if (!Option.contains(actualRepository, expectedRepository)) {
      return yield* fail(
        `${gitmodulesPath} does not map ${submoduleName} to ${expectedRepository}`
      );
    }
  });

const requireOneLockedPackage = (
  lock: typeof CargoLockSchema.Type,
  name: "llama-cpp-2" | "llama-cpp-sys-2"
) => {
  const matches = lock.package.filter((entry) => entry.name === name);
  if (matches.length !== 1) {
    return Effect.fail(
      fail(`Cargo.lock must contain exactly one local ${name} package`)
    );
  }
  const entry = Option.fromNullable(matches[0]);
  return Option.match(entry, {
    onNone: () => Effect.fail(fail(`Cargo.lock is missing ${name}`)),
    onSome: Effect.succeed,
  });
};

export const verifyNativePin = Effect.fn("verifyNativePin")(function* () {
  const pin = yield* decodeToml(
    NativePinSchema,
    resolve(inferenceRoot, "native-pin.toml")
  );
  const manifest = yield* decodeToml(
    CargoManifestSchema,
    resolve(inferenceRoot, "Cargo.toml")
  );
  const lock = yield* decodeToml(
    CargoLockSchema,
    resolve(inferenceRoot, "Cargo.lock")
  );

  const bindingsCheckout = yield* requirePathInside(
    inferenceRoot,
    pin.llama_cpp_rs.checkout_path
  );
  const bindingsCrate = yield* requirePathInside(
    inferenceRoot,
    pin.llama_cpp_rs.crate_path
  );
  const llamaCppCheckout = yield* requirePathInside(
    inferenceRoot,
    pin.llama_cpp.checkout_path
  );

  yield* requireExists(resolve(bindingsCrate, "Cargo.toml"));
  yield* requireExists(resolve(llamaCppCheckout, "include/llama.h"));
  yield* requireExists(resolve(llamaCppCheckout, "tools/server/CMakeLists.txt"));
  yield* requireExists(resolve(llamaCppCheckout, "tools/llama-bench/CMakeLists.txt"));

  const dependency = manifest.workspace.dependencies["llama-cpp-2"];
  if (dependency.path !== pin.llama_cpp_rs.crate_path) {
    return yield* fail("Cargo path does not match native-pin.toml");
  }
  if (Option.isSome(dependency.git) || Option.isSome(dependency.rev)) {
    return yield* fail("llama-cpp-2 must be a path-only Cargo dependency");
  }

  const bindingManifest = yield* decodeToml(
    BindingManifestSchema,
    resolve(bindingsCrate, "Cargo.toml")
  );
  const sysManifest = yield* decodeToml(
    SysManifestSchema,
    resolve(bindingsCheckout, "llama-cpp-sys-2/Cargo.toml")
  );
  if (
    bindingManifest.package.version !== pin.llama_cpp_rs.release ||
    sysManifest.package.version !== pin.llama_cpp_rs.release
  ) {
    return yield* fail("Checked-out binding crate versions do not match native-pin.toml");
  }
  if (
    bindingManifest.package.repository !== pin.llama_cpp_rs.upstream_repository ||
    sysManifest.package.repository !== pin.llama_cpp_rs.upstream_repository
  ) {
    return yield* fail("Binding package provenance does not match native-pin.toml");
  }
  if (
    bindingManifest.dependencies["llama-cpp-sys-2"].path !==
      "../llama-cpp-sys-2" ||
    bindingManifest.dependencies["llama-cpp-sys-2"].version !==
      pin.llama_cpp_rs.release
  ) {
    return yield* fail("llama-cpp-2 does not consume its adjacent pinned sys crate");
  }

  const lockedBinding = yield* requireOneLockedPackage(lock, "llama-cpp-2");
  const lockedSys = yield* requireOneLockedPackage(lock, "llama-cpp-sys-2");
  if (
    lockedBinding.version !== pin.llama_cpp_rs.release ||
    lockedSys.version !== pin.llama_cpp_rs.release
  ) {
    return yield* fail("Cargo.lock binding versions do not match native-pin.toml");
  }
  if (Option.isSome(lockedBinding.source) || Option.isSome(lockedSys.source)) {
    return yield* fail("Cargo.lock still resolves llama-cpp-rs through a remote source");
  }

  yield* verifySubmodule(
    resolve(repositoryRoot, ".gitmodules"),
    "inference/native/llama-cpp-rs",
    "inference/native/llama-cpp-rs",
    pin.llama_cpp_rs.repository
  );
  yield* verifySubmodule(
    resolve(bindingsCheckout, ".gitmodules"),
    "llama-cpp-sys-2/llama.cpp",
    "llama-cpp-sys-2/llama.cpp",
    pin.llama_cpp.repository
  );

  const bindingRevision = yield* readCheckoutRevision(bindingsCheckout);
  const llamaCppRevision = yield* readCheckoutRevision(llamaCppCheckout);
  if (bindingRevision !== pin.llama_cpp_rs.revision) {
    return yield* fail(
      `Checked-out llama-cpp-rs is ${bindingRevision}, expected ${pin.llama_cpp_rs.revision}`
    );
  }
  if (llamaCppRevision !== pin.llama_cpp.revision) {
    return yield* fail(
      `Checked-out llama.cpp is ${llamaCppRevision}, expected ${pin.llama_cpp.revision}`
    );
  }

  return pin;
});

if (import.meta.main) {
  const pin = await Effect.runPromise(verifyNativePin());
  console.log(
    `llama-cpp-rs ${pin.llama_cpp_rs.release} ${pin.llama_cpp_rs.revision.slice(0, 12)} -> llama.cpp ${pin.llama_cpp.revision.slice(0, 12)} (local source verified)`
  );
}
