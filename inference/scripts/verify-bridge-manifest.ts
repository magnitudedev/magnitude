import { Data, Effect, Schema } from "effect";
import { isAbsolute, relative, resolve, sep } from "node:path";

class BindingSurfaceError extends Data.TaggedError("BindingSurfaceError")<{
  readonly message: string;
}> {}

const Revision = Schema.String.pipe(Schema.pattern(/^[0-9a-f]{40}$/));

const Surface = Schema.Struct({
  id: Schema.String,
  upstreamFiles: Schema.Array(Schema.String),
  upstreamSymbols: Schema.Array(Schema.String),
  bridgeSymbols: Schema.Array(Schema.String),
  rustModules: Schema.Array(Schema.String),
  rustSymbols: Schema.Array(Schema.String),
});

const BindingSurfaces = Schema.Struct({
  schemaVersion: Schema.Literal(1),
  bindingsRevision: Revision,
  llamaCppRevision: Revision,
  surfaces: Schema.Array(Surface),
});

const NativePin = Schema.Struct({
  schema_version: Schema.Literal(1),
  llama_cpp_rs: Schema.Struct({ revision: Revision }),
  llama_cpp: Schema.Struct({ revision: Revision }),
});

const inferenceRoot = resolve(import.meta.dirname, "..");
const bindingsRoot = resolve(inferenceRoot, "native/llama-cpp-rs");
const sysRoot = resolve(bindingsRoot, "llama-cpp-sys-2");
const llamaRoot = resolve(sysRoot, "llama.cpp");
const inventoryPath = resolve(
  inferenceRoot,
  "parity/upstream/binding-surfaces.json"
);
const pinPath = resolve(inferenceRoot, "native-pin.toml");
const fail = (message: string) => new BindingSurfaceError({ message });

const read = (path: string) =>
  Effect.promise(() => Bun.file(path).exists()).pipe(
    Effect.flatMap((present) =>
      present
        ? Effect.tryPromise({
            try: () => Bun.file(path).text(),
            catch: (cause) => fail(`Unable to read ${path}: ${String(cause)}`),
          })
        : Effect.fail(fail(`Missing ${path}`))
    )
  );

const decodeJson = <A, I>(schema: Schema.Schema<A, I>, path: string) =>
  read(path).pipe(
    Effect.flatMap(Schema.decode(Schema.parseJson(schema))),
    Effect.mapError((cause) => fail(`Invalid ${path}: ${String(cause)}`))
  );

const decodeNativePin = (path: string) =>
  read(path).pipe(
    Effect.map((contents) => Bun.TOML.parse(contents)),
    Effect.flatMap(Schema.decodeUnknown(NativePin)),
    Effect.mapError((cause) => fail(`Invalid ${path}: ${String(cause)}`))
  );

const pathWithin = (root: string, path: string, owner: string) => {
  if (isAbsolute(path)) {
    return Effect.fail(fail(`${owner}: path must be relative: ${path}`));
  }
  const absolute = resolve(root, path);
  const fromRoot = relative(root, absolute);
  if (
    fromRoot === ".." ||
    fromRoot.startsWith(`..${sep}`) ||
    isAbsolute(fromRoot)
  ) {
    return Effect.fail(fail(`${owner}: path escapes its source root: ${path}`));
  }
  return Effect.succeed(absolute);
};

const requireNonEmpty = <T>(
  surface: string,
  label: string,
  values: ReadonlyArray<T>
) =>
  values.length > 0
    ? Effect.void
    : Effect.fail(fail(`${surface}: ${label} must not be empty`));

const requireUnique = (
  surface: string,
  label: string,
  values: ReadonlyArray<string>
) => {
  const duplicates = [
    ...new Set(
      values.filter((value, index) => values.indexOf(value) !== index)
    ),
  ];
  return duplicates.length === 0
    ? Effect.void
    : Effect.fail(
        fail(`${surface}: duplicate ${label}: ${duplicates.sort().join(", ")}`)
      );
};

const requirePopulatedStrings = (
  surface: string,
  label: string,
  values: ReadonlyArray<string>
) => {
  const emptyIndexes = values.flatMap((value, index) =>
    value.trim().length === 0 ? [index] : []
  );
  return emptyIndexes.length === 0
    ? Effect.void
    : Effect.fail(
        fail(
          `${surface}: ${label} contains empty entries at indexes ${emptyIndexes.join(", ")}`
        )
      );
};

const escapeRegExp = (value: string) =>
  value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const containsSymbol = (contents: string, symbol: string) =>
  new RegExp(`\\b${escapeRegExp(symbol)}\\b`).test(contents);

const exportedBridgeFunctions = (header: string) => {
  const withoutComments = header
    .replaceAll(/\/\*[\s\S]*?\*\//g, " ")
    .replaceAll(/\/\/[^\n]*/g, " ");
  const blocks: Array<string> = [];
  const blockStart = /extern\s+"C"\s*\{/g;
  for (const match of withoutComments.matchAll(blockStart)) {
    const opening = (match.index ?? 0) + match[0].length - 1;
    let depth = 1;
    for (let index = opening + 1; index < withoutComments.length; index += 1) {
      const character = withoutComments[index];
      if (character === "{") depth += 1;
      if (character === "}") depth -= 1;
      if (depth === 0) {
        blocks.push(withoutComments.slice(opening + 1, index));
        break;
      }
    }
  }

  const names = new Set<string>();
  const declaration = /([^;{}]*\b(llama_rs_[A-Za-z0-9_]+)\s*\([^;{}]*\)\s*;)/g;
  for (const match of blocks.join("\n").matchAll(declaration)) {
    const statement = match[1] ?? "";
    const name = match[2];
    if (name !== undefined && !/\b(?:static|typedef)\b/.test(statement)) {
      names.add(name);
    }
  }
  return names;
};

const program = Effect.gen(function* () {
  const [pin, inventory] = yield* Effect.all([
    decodeNativePin(pinPath),
    decodeJson(BindingSurfaces, inventoryPath),
  ]);

  if (
    inventory.bindingsRevision !== pin.llama_cpp_rs.revision ||
    inventory.llamaCppRevision !== pin.llama_cpp.revision
  ) {
    return yield* fail(
      "Binding-surface revisions do not match inference/native-pin.toml"
    );
  }
  yield* requireNonEmpty("inventory", "surfaces", inventory.surfaces);

  const wrapperHeaders = yield* Effect.tryPromise({
    try: async () => {
      const paths: Array<string> = [];
      for await (const path of new Bun.Glob("wrapper*.h").scan({
        cwd: sysRoot,
        onlyFiles: true,
      })) {
        paths.push(path);
      }
      return paths.sort();
    },
    catch: (cause) =>
      fail(`Unable to enumerate checked-in wrapper headers: ${String(cause)}`),
  });
  yield* requireNonEmpty(
    "bindings",
    "checked-in wrapper headers",
    wrapperHeaders
  );
  const bridgeDeclarations = new Set<string>();
  for (const path of wrapperHeaders) {
    const header = yield* read(resolve(sysRoot, path));
    for (const symbol of exportedBridgeFunctions(header)) {
      bridgeDeclarations.add(symbol);
    }
  }

  const surfaceIds = new Set<string>();
  const assignedBridgeSymbols = new Map<string, string>();
  let upstreamSymbolCount = 0;
  let rustSymbolCount = 0;

  for (const surface of inventory.surfaces) {
    if (surface.id.trim().length === 0) {
      return yield* fail("Binding-surface id must not be empty");
    }
    if (surfaceIds.has(surface.id)) {
      return yield* fail(`Duplicate binding-surface id ${surface.id}`);
    }
    surfaceIds.add(surface.id);

    for (const [label, values] of [
      ["upstream files", surface.upstreamFiles],
      ["upstream symbols", surface.upstreamSymbols],
      ["bridge symbols", surface.bridgeSymbols],
      ["Rust modules", surface.rustModules],
      ["Rust symbols", surface.rustSymbols],
    ] as const) {
      yield* requireUnique(surface.id, label, values);
      yield* requirePopulatedStrings(surface.id, label, values);
    }
    yield* requireNonEmpty(surface.id, "upstreamFiles", surface.upstreamFiles);
    yield* requireNonEmpty(surface.id, "upstreamSymbols", surface.upstreamSymbols);
    yield* requireNonEmpty(surface.id, "rustModules", surface.rustModules);
    yield* requireNonEmpty(surface.id, "rustSymbols", surface.rustSymbols);

    const upstream = (
      yield* Effect.forEach(surface.upstreamFiles, (path) =>
        pathWithin(llamaRoot, path, surface.id).pipe(Effect.flatMap(read))
      )
    ).join("\n");
    for (const symbol of surface.upstreamSymbols) {
      if (!containsSymbol(upstream, symbol)) {
        return yield* fail(`${surface.id}: missing upstream symbol ${symbol}`);
      }
      upstreamSymbolCount += 1;
    }

    for (const symbol of surface.bridgeSymbols) {
      const previous = assignedBridgeSymbols.get(symbol);
      if (previous !== undefined) {
        return yield* fail(
          `${symbol} is assigned to both ${previous} and ${surface.id}`
        );
      }
      assignedBridgeSymbols.set(symbol, surface.id);
      if (!bridgeDeclarations.has(symbol)) {
        return yield* fail(
          `${surface.id}: missing C bridge declaration ${symbol}`
        );
      }
    }

    const rust = (
      yield* Effect.forEach(surface.rustModules, (path) =>
        pathWithin(bindingsRoot, path, surface.id).pipe(Effect.flatMap(read))
      )
    ).join("\n");
    for (const symbol of surface.rustSymbols) {
      if (!containsSymbol(rust, symbol)) {
        return yield* fail(`${surface.id}: missing safe Rust symbol ${symbol}`);
      }
      rustSymbolCount += 1;
    }
  }

  console.log(
    `Verified ${inventory.surfaces.length} pinned binding surfaces: ` +
      `${upstreamSymbolCount} upstream symbols, ` +
      `${assignedBridgeSymbols.size} C bridge declarations, and ` +
      `${rustSymbolCount} safe Rust symbols. Behavioral parity is checked separately.`
  );
});

Effect.runPromise(program).catch((error) => {
  console.error(error);
  process.exit(1);
});
