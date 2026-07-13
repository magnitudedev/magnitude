import * as FileSystem from "@effect/platform/FileSystem";
import type { PlatformError } from "@effect/platform/Error";
import * as NodePath from "node:path";
import { randomUUID } from "node:crypto";
import { Effect, Schema } from "effect";

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
}

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
      })
    )(value).pipe(
      Effect.mapError(
        (error) =>
          new StructuredFileEncodeFailed({ path, reason: String(error) })
      )
    );
    if (options?.appendNewline !== false && !content.endsWith("\n")) {
      content += "\n";
    }

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
          if (options?.mode !== undefined) {
            yield* fs.chmod(tmpPath, options.mode);
          }
          yield* fs.rename(tmpPath, path);
        }),
      (tmpPath) =>
        fs
          .remove(tmpPath, { force: true })
          .pipe(Effect.catchAll(() => Effect.void))
    );
  });
