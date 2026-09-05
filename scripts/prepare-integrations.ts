import * as FileSystem from "@effect/platform/FileSystem";
import { BunContext, BunRuntime } from "@effect/platform-bun";
import { Console, Effect, Schema } from "effect";
import { resolve } from "node:path";
import { canonical } from "@magnitudedev/utils/canonical-key";
import {
  packPlugin,
  requireNpm,
  verifyPluginArtifact,
  PluginArtifactError,
} from "../packages/release/src/plugin-artifacts";
import {
  prepareRelease,
  readPreparedRelease,
} from "../packages/release/scripts/prepare-release";
import { PreparedReleaseSchema } from "../packages/release/src/release-plan";

const root = resolve(import.meta.dir, "..");
const program = Effect.gen(function* () {
  const directory = resolve(process.argv[2] ?? "release/integration-candidate");
  yield* prepareRelease("verify");
  const plan = yield* readPreparedRelease;
  const fs = yield* FileSystem.FileSystem;
  yield* fs.makeDirectory(directory, { recursive: true });
  for (const { artifact, publish } of plan.plugins) {
    const file = `${directory}/${artifact.filename}`;
    if (yield* fs.exists(file)) {
      yield* verifyPluginArtifact(artifact, directory);
      continue;
    }
    if (publish) {
      const actual = yield* packPlugin(
        resolve(root, "integrations/pi"),
        directory
      );
      if (canonical(actual) !== canonical(artifact))
        return yield* new PluginArtifactError({
          message:
            "Packed plugin differs from the prepared artifact; refresh the release plan",
        });
    } else {
      // Unchanged means reuse the already-published bytes, not rebuild that version.
      const url = yield* requireNpm(
        [
          "view",
          `${artifact.name}@${artifact.version}`,
          "dist.tarball",
          "--json",
        ],
        root
      ).pipe(
        Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(Schema.String)))
      );
      const bytes = yield* Effect.tryPromise({
        try: async (signal) => {
          const response = await fetch(url, { signal });
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          return new Uint8Array(await response.arrayBuffer());
        },
        catch: (error) =>
          new PluginArtifactError({
            message: `Cannot download pinned plugin: ${String(error)}`,
          }),
      }).pipe(Effect.timeout("2 minutes"));
      yield* fs.writeFile(file, bytes);
      yield* verifyPluginArtifact(artifact, directory);
    }
  }
  yield* fs.writeFileString(
    `${directory}/release-plan.json`,
    yield* Schema.encode(Schema.parseJson(PreparedReleaseSchema, { space: 2 }))(
      plan
    )
  );
  yield* Console.log(
    `Prepared immutable plugin artifacts at ${directory}. Acceptance and publication consume these files.`
  );
});
if (import.meta.main)
  BunRuntime.runMain(program.pipe(Effect.provide(BunContext.layer)));
