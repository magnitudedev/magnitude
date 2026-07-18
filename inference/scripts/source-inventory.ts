import { Data, Effect } from "effect";
import { resolve } from "node:path";

export class SourceInventoryError extends Data.TaggedError(
  "SourceInventoryError"
)<{
  readonly message: string;
}> {}

export interface SourceInventoryEntry {
  readonly sourcePath: string;
  readonly bytes: number;
  readonly sha256: string;
}

export interface SourceInventory {
  readonly algorithm: "sha256-path-size-content-v1";
  readonly sha256: string;
  readonly fileCount: number;
  readonly totalBytes: number;
  readonly excludedDirectoryNames: ReadonlyArray<string>;
  readonly entries: ReadonlyArray<SourceInventoryEntry>;
}

export interface SourceInventoryOptions {
  /** Directory base names excluded at every depth. */
  readonly excludedDirectoryNames?: ReadonlyArray<string>;
}

export const defaultExcludedDirectoryNames = [
  ".git",
  "node_modules",
  ".cache",
  "target",
] as const;

const fail = (cause: unknown) =>
  new SourceInventoryError({
    message: cause instanceof Error ? cause.message : String(cause),
  });

/**
 * Produce a deterministic content inventory without consulting Git metadata.
 *
 * The aggregate hashes each sorted relative path, byte length, and per-file SHA-256. VCS metadata
 * and generated dependency/cache directories are excluded by name; all other regular files are
 * conservatively treated as native build inputs.
 */
export const sourceInventory = Effect.fn("sourceInventory")(function* (
  sourceDirectory: string,
  options: SourceInventoryOptions = {}
) {
  const excludedDirectoryNames = [
    ...(options.excludedDirectoryNames ?? defaultExcludedDirectoryNames),
  ];
  if (
    excludedDirectoryNames.length === 0 ||
    new Set(excludedDirectoryNames).size !== excludedDirectoryNames.length ||
    excludedDirectoryNames.some(
      (name) =>
        name.length === 0 ||
        name === "." ||
        name === ".." ||
        name.includes("/") ||
        name.includes("\\")
    )
  ) {
    return yield* new SourceInventoryError({
      message:
        "excludedDirectoryNames must contain unique, non-empty directory base names",
    });
  }
  const excluded = new Set(excludedDirectoryNames);
  const paths = yield* Effect.tryPromise({
    try: async () => {
      const collected: Array<string> = [];
      const glob = new Bun.Glob("**/*");
      for await (const path of glob.scan({
        cwd: sourceDirectory,
        dot: true,
        followSymlinks: false,
        onlyFiles: true,
      })) {
        const segments = path.split("/");
        // The contract excludes directory base names, not a regular file that
        // happens to share one of those names. This matches the Rust verifier.
        if (segments.slice(0, -1).some((segment) => excluded.has(segment))) {
          continue;
        }
        collected.push(path);
      }
      return collected.sort((left, right) =>
        Buffer.compare(Buffer.from(left, "utf8"), Buffer.from(right, "utf8"))
      );
    },
    catch: fail,
  });
  const entries = yield* Effect.forEach(
    paths,
    (sourcePath) =>
      Effect.tryPromise({
        try: async () => {
          const file = Bun.file(resolve(sourceDirectory, sourcePath));
          const hasher = new Bun.CryptoHasher("sha256");
          hasher.update(await file.arrayBuffer());
          return {
            sourcePath,
            bytes: file.size,
            sha256: hasher.digest("hex"),
          } satisfies SourceInventoryEntry;
        },
        catch: fail,
      }),
    { concurrency: 8 }
  );
  const aggregate = new Bun.CryptoHasher("sha256");
  let totalBytes = 0;
  for (const entry of entries) {
    aggregate.update(`${entry.sourcePath}\0${entry.bytes}\0${entry.sha256}\n`);
    totalBytes += entry.bytes;
  }
  return {
    algorithm: "sha256-path-size-content-v1",
    sha256: aggregate.digest("hex"),
    fileCount: entries.length,
    totalBytes,
    excludedDirectoryNames,
    entries,
  } satisfies SourceInventory;
});
