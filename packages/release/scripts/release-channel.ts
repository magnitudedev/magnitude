import * as FileSystem from "@effect/platform/FileSystem";
import { Effect, Option, Schema } from "effect";
import { prerelease } from "semver";
import { resolve } from "node:path";

/** Changesets' `.changeset/pre.json`: present while a prerelease series is open or being exited. */
export const PreStateSchema = Schema.Struct({
  mode: Schema.Literal("pre", "exit"),
  tag: Schema.String,
});
export const isPrereleaseVersion = (version: string): boolean =>
  prerelease(version) !== null;

/** The npm dist-tag a version publishes under: its prerelease identifier, else `latest`. */
export const publishTag = (version: string): string =>
  String(prerelease(version)?.[0] ?? "latest");

/** The open prerelease tag, if Changesets is in pre mode. `pre exit` counts as stable. */
export const readPrereleaseTag = (
  root = resolve(import.meta.dir, "../../..")
) =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const path = `${root}/.changeset/pre.json`;
    if (!(yield* fs.exists(path))) return Option.none<string>();
    const state = yield* fs
      .readFileString(path)
      .pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(PreStateSchema))));
    return state.mode === "pre" ? Option.some(state.tag) : Option.none<string>();
  });
