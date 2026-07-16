import * as FileSystem from "@effect/platform/FileSystem";
import type { PlatformError } from "@effect/platform/Error";
import * as NodePath from "node:path";
import { randomUUID } from "node:crypto";
import { Effect, Either, ParseResult, Schema, SchemaAST } from "effect";

export class StructuredFileInvalid extends Schema.TaggedError<StructuredFileInvalid>()(
  "StructuredFileInvalid",
  {
    path: Schema.String,
    reason: Schema.String,
  }
) {}

export class StructuredFileEncodeFailed extends Schema.TaggedError<StructuredFileEncodeFailed>()(
  "StructuredFileEncodeFailed",
  {
    path: Schema.String,
    reason: Schema.String,
  }
) {}

export type StructuredFileRead<A> =
  | { readonly _tag: "Missing" }
  | { readonly _tag: "Present"; readonly value: A }
  | { readonly _tag: "Invalid"; readonly error: StructuredFileInvalid };

export interface StructuredFileRecoveryIssue {
  readonly tag: string;
  readonly path: ReadonlyArray<PropertyKey>;
  readonly message: string;
}

export interface StructuredFileRecovery {
  readonly recovered: boolean;
  readonly resetRoot: boolean;
  readonly attempts: number;
  readonly issues: ReadonlyArray<StructuredFileRecoveryIssue>;
  readonly removedPaths: ReadonlyArray<ReadonlyArray<PropertyKey>>;
}

export type RecoverableStructuredFileRead<A> =
  | { readonly _tag: "Missing" }
  | {
      readonly _tag: "Present";
      readonly value: A;
      readonly originalText: string;
      readonly recovery: StructuredFileRecovery;
    }
  | {
      readonly _tag: "Malformed";
      readonly originalText: string;
      readonly reason: string;
    }
  | {
      readonly _tag: "Unrecoverable";
      readonly originalText: string;
      readonly reason: string;
    };

const forbiddenPathKeys = new Set<PropertyKey>([
  "__proto__",
  "prototype",
  "constructor",
]);

const isObject = (value: unknown): value is Record<PropertyKey, unknown> =>
  typeof value === "object" && value !== null;

const samePath = (
  left: ReadonlyArray<PropertyKey>,
  right: ReadonlyArray<PropertyKey>
): boolean =>
  left.length === right.length && left.every((part, index) => part === right[index]);

const isStrictPrefix = (
  prefix: ReadonlyArray<PropertyKey>,
  path: ReadonlyArray<PropertyKey>
): boolean =>
  prefix.length < path.length && prefix.every((part, index) => part === path[index]);

const deepestUniquePaths = (
  paths: ReadonlyArray<ReadonlyArray<PropertyKey>>
): ReadonlyArray<ReadonlyArray<PropertyKey>> => {
  const unique = paths.filter(
    (path, index) => paths.findIndex((candidate) => samePath(candidate, path)) === index
  );
  return unique.filter(
    (path) => !unique.some((candidate) => isStrictPrefix(path, candidate))
  );
};

const removeAtPath = (
  root: unknown,
  path: ReadonlyArray<PropertyKey>
): ReadonlyArray<PropertyKey> | null => {
  if (path.length === 0 || path.some((part) => forbiddenPathKeys.has(part))) return null;

  const nodes: Array<{ readonly parent: Record<PropertyKey, unknown> | unknown[]; readonly key: PropertyKey }> = [];
  let current: unknown = root;
  for (const key of path) {
    if (!isObject(current)) break;
    const parent = current as Record<PropertyKey, unknown> | unknown[];
    const exists = Array.isArray(parent)
      ? typeof key === "number" && Number.isInteger(key) && key >= 0 && key < parent.length
      : Object.prototype.hasOwnProperty.call(parent, key);
    if (!exists) break;
    nodes.push({ parent, key });
    current = (parent as Record<PropertyKey, unknown>)[key];
  }

  const target = nodes.at(-1);
  if (!target) return null;
  if (Array.isArray(target.parent)) {
    if (typeof target.key !== "number") return null;
    target.parent.splice(target.key, 1);
  } else {
    delete target.parent[target.key];
  }
  return path.slice(0, nodes.length);
};

const countJsonNodes = (value: unknown): number => {
  if (!isObject(value)) return 1;
  return 1 + Object.keys(value).reduce(
    (total, key) => total + countJsonNodes((value as Record<string, unknown>)[key]),
    0
  );
};

/**
 * Decode a JSON document while salvaging valid siblings. Recovery removes the
 * deepest failing paths reported by Effect Schema and retries. Required fields
 * naturally cause a later pass to remove their nearest existing parent.
 */
export const readRecoverableStructuredFile = <A, I, R>(
  path: string,
  schema: Schema.Schema<A, I, R>,
  options: {
    readonly rootDefault: () => A;
    readonly preserveExcessProperties?: boolean;
  }
): Effect.Effect<RecoverableStructuredFileRead<A>, PlatformError, FileSystem.FileSystem | R> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const text = yield* fs.readFileString(path).pipe(
      Effect.catchTag("SystemError", (error) =>
        error.reason === "NotFound" ? Effect.succeed(null) : Effect.fail(error)
      )
    );
    if (text === null) return { _tag: "Missing" } as const;

    const parsed = yield* Effect.either(
      Schema.decodeUnknown(Schema.parseJson(Schema.Unknown))(text)
    );
    if (Either.isLeft(parsed)) {
      return {
        _tag: "Malformed",
        originalText: text,
        reason: String(parsed.left),
      } as const;
    }

    const candidate = parsed.right;
    const parseOptions: SchemaAST.ParseOptions = {
      errors: "all",
      onExcessProperty: options.preserveExcessProperties === false ? "ignore" : "preserve",
    };
    const issues: StructuredFileRecoveryIssue[] = [];
    const removedPaths: Array<ReadonlyArray<PropertyKey>> = [];
    const maxAttempts = countJsonNodes(candidate) + 1;
    let attempts = 0;

    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
      attempts = attempt;
      const decoded = yield* Effect.either(Schema.decodeUnknown(schema, parseOptions)(candidate));
      if (Either.isRight(decoded)) {
        return {
          _tag: "Present",
          value: decoded.right,
          originalText: text,
          recovery: {
            recovered: removedPaths.length > 0,
            resetRoot: false,
            attempts: attempt,
            issues,
            removedPaths,
          },
        } as const;
      }

      const formatted = yield* ParseResult.ArrayFormatter.formatError(decoded.left);
      issues.push(...formatted.map((issue) => ({
        tag: issue._tag,
        path: issue.path,
        message: issue.message,
      })));
      const paths = deepestUniquePaths(formatted.map((issue) => issue.path));
      if (paths.some((issuePath) => issuePath.length === 0 || issuePath.some((part) => forbiddenPathKeys.has(part)))) {
        break;
      }

      // Removing higher array indexes first prevents index shifts from changing
      // which element a later path identifies.
      const ordered = [...paths].sort((left, right) => {
        const commonLength = Math.min(left.length, right.length);
        for (let index = 0; index < commonLength; index += 1) {
          const leftPart = left[index];
          const rightPart = right[index];
          if (leftPart === rightPart) continue;
          if (typeof leftPart === "number" && typeof rightPart === "number") {
            return rightPart - leftPart;
          }
          break;
        }
        return right.length - left.length;
      });
      let madeProgress = false;
      for (const issuePath of ordered) {
        const removed = removeAtPath(candidate, issuePath);
        if (!removed) continue;
        madeProgress = true;
        if (!removedPaths.some((existing) => samePath(existing, removed))) removedPaths.push(removed);
      }
      if (!madeProgress) break;
    }

    const rootDefault = options.rootDefault();
    const validatedDefault = yield* Effect.either(Schema.validate(schema)(rootDefault));
    if (Either.isLeft(validatedDefault)) {
      return {
        _tag: "Unrecoverable",
        originalText: text,
        reason: `Invalid root default: ${String(validatedDefault.left)}`,
      } as const;
    }
    return {
      _tag: "Present",
      value: validatedDefault.right,
      originalText: text,
      recovery: {
        recovered: true,
        resetRoot: true,
        attempts,
        issues,
        removedPaths,
      },
    } as const;
  });

/**
 * Read and decode a structured JSON file without allowing malformed external
 * bytes to become an Effect defect. Missing, invalid, and inaccessible remain
 * distinct states.
 */
export const readStructuredFile = <A, I>(
  path: string,
  schema: Schema.Schema<A, I>
): Effect.Effect<StructuredFileRead<A>, PlatformError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const text = yield* fs
      .readFileString(path)
      .pipe(
        Effect.catchTag("SystemError", (error) =>
          error.reason === "NotFound"
            ? Effect.succeed(null)
            : Effect.fail(error)
        )
      );
    if (text === null) return { _tag: "Missing" } as const;

    return yield* Schema.decodeUnknown(Schema.parseJson(schema))(text).pipe(
      Effect.map((value) => ({ _tag: "Present", value } as const)),
      Effect.catchAll((error) =>
        Effect.succeed({
          _tag: "Invalid",
          error: new StructuredFileInvalid({ path, reason: String(error) }),
        } as const)
      )
    );
  });

export interface StructuredFileWriteOptions {
  readonly mode?: number;
  readonly space?: number;
  readonly appendNewline?: boolean;
  readonly parseOptions?: SchemaAST.ParseOptions;
}

export const writeTextFileAtomic = (
  path: string,
  content: string,
  options?: { readonly mode?: number }
): Effect.Effect<void, PlatformError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const directory = NodePath.dirname(path);
    const temporaryPath = NodePath.join(
      directory,
      `.${NodePath.basename(path)}.${process.pid}.${randomUUID()}.tmp`
    );
    yield* fs.makeDirectory(directory, { recursive: true });
    yield* Effect.acquireUseRelease(
      Effect.succeed(temporaryPath),
      (tmpPath) =>
        Effect.gen(function* () {
          yield* fs.writeFileString(
            tmpPath,
            content,
            options?.mode === undefined ? undefined : { mode: options.mode }
          );
          if (options?.mode !== undefined) yield* fs.chmod(tmpPath, options.mode);
          yield* fs.rename(tmpPath, path);
        }),
      (tmpPath) => fs.remove(tmpPath, { force: true }).pipe(Effect.catchAll(() => Effect.void))
    );
  });

/**
 * Encode completely before opening a file, then publish with one same-directory
 * rename. Interruption cleans the uniquely named temporary file.
 */
export const writeStructuredFileAtomic = <A, I>(
  path: string,
  schema: Schema.Schema<A, I>,
  value: A,
  options?: StructuredFileWriteOptions
): Effect.Effect<
  void,
  PlatformError | StructuredFileEncodeFailed,
  FileSystem.FileSystem
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    let content = yield* Schema.encodeUnknown(
      Schema.parseJson(schema, {
        space: options?.space ?? 2,
      }),
      options?.parseOptions
    )(value).pipe(
      Effect.mapError(
        (error) =>
          new StructuredFileEncodeFailed({ path, reason: String(error) })
      )
    );
    if (options?.appendNewline !== false && !content.endsWith("\n")) {
      content += "\n";
    }

    yield* writeTextFileAtomic(path, content, { mode: options?.mode });
  });
