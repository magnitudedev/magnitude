import * as FileSystem from "@effect/platform/FileSystem";
import { BunContext, BunRuntime } from "@effect/platform-bun";
import { Config, Effect, Schema } from "effect";
import { resolve } from "node:path";
import { prerelease } from "semver";
import {
  artifactIntegrity,
  awaitPublishedIntegrity,
  publishedPluginIntegrity,
  requireNpm,
  verifyPublishedPlugins,
  PluginArtifactError,
} from "../src/plugin-artifacts";
import releasePlan from "../release-plan.json";
import {
  PreparedReleaseSchema,
  validatePreparedRelease,
} from "../src/release-plan";
import { validateNpmCandidate } from "./prepare-npm";

// Deliberately not `changeset publish`: that would independently pack and publish
// every public workspace, bypassing plugin-first acceptance and exact artifacts.
const program = Effect.gen(function* () {
  const argument = process.argv[2];
  if (!argument || argument.startsWith("-"))
    return yield* new PluginArtifactError({
      message: "Usage: bun run publish <accepted-cli-tarball>",
    });
  const tarball = resolve(argument);
  const version = yield* Config.string("MAGNITUDE_RELEASE_VERSION");
  const expected = yield* Config.string("MAGNITUDE_EXPECTED_NPM_INTEGRITY");
  const fs = yield* FileSystem.FileSystem;
  const source = yield* fs
    .readFileString(resolve(import.meta.dir, "../../launcher/package.json"))
    .pipe(
      Effect.flatMap(
        Schema.decodeUnknown(
          Schema.parseJson(
            Schema.Struct({
              name: Schema.Literal("@magnitudedev/cli"),
              version: Schema.String,
            })
          )
        )
      )
    );
  if (
    source.version !== version ||
    artifactIntegrity(yield* fs.readFile(tarball)) !== expected
  )
    return yield* new PluginArtifactError({
      message: "CLI candidate identity or integrity mismatch",
    });
  yield* Effect.tryPromise(() => validateNpmCandidate(tarball));
  const cwd = resolve(import.meta.dir, "../../..");
  // Every channel pins plugins; a prerelease CLI pins prerelease plugins.
  const plan = yield* Schema.decodeUnknown(PreparedReleaseSchema)(
    releasePlan
  ).pipe(Effect.flatMap(validatePreparedRelease));
  yield* verifyPublishedPlugins(
    plan.plugins.map(({ artifact }) => artifact),
    cwd
  );
  const existing = yield* publishedPluginIntegrity(source.name, version, cwd);
  if (existing !== null && existing !== expected)
    return yield* new PluginArtifactError({
      message: `${source.name}@${version} is already published with integrity ${existing}, not the accepted ${expected}; a registry version is immutable`,
    });
  const tag = String(prerelease(version)?.[0] ?? "latest");
  if (existing === null)
    yield* requireNpm(
      [
        "publish",
        tarball,
        "--access",
        "public",
        "--ignore-scripts",
        "--tag",
        tag,
      ],
      cwd
    );
  // The registry lags a publish; prove it serves the accepted bytes before reporting success.
  yield* awaitPublishedIntegrity(source.name, version, expected, cwd);
});
if (import.meta.main)
  BunRuntime.runMain(program.pipe(Effect.provide(BunContext.layer)));
