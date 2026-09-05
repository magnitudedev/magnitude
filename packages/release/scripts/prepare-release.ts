import * as Command from "@effect/platform/Command";
import * as FileSystem from "@effect/platform/FileSystem";
import { BunContext, BunRuntime } from "@effect/platform-bun";
import { Console, Effect, Option, Schema } from "effect";
import { inc } from "semver";
import { resolve } from "node:path";
import { canonical } from "@magnitudedev/utils/canonical-key";
import { JsonValueSchema } from "@magnitudedev/utils/schema";
import { verifyPluginContent } from "@magnitudedev/release/plugin-content";
import {
  generateVersionFiles,
  readGeneratedRpcVersion,
} from "@magnitudedev/version/scripts/generate-version";
import {
  packPlugin,
  publishedPlugin,
  publishedPluginIntegrity,
  requireNpm,
} from "../src/plugin-artifacts";
import {
  CLI_PACKAGE_NAME,
  declaredChangesetReleases,
  derivedChangeset,
} from "./derived-changeset";
import { rpcFingerprint } from "./rpc-fingerprint";
import { readPublicBaseline } from "./public-baseline";
import { readPrereleaseTag } from "./release-channel";
import { reconcilePluginChangelog } from "./plugin-changelog";
import {
  allocateRevision,
  allocateRpcVersion,
  isAwaitingPublication,
  planPlugin,
  PreparedReleaseSchema,
  ReleasePreparationFailed,
  validatePreparedRelease,
  type PreparedRelease,
} from "../src/release-plan";

const root = resolve(import.meta.dir, "../../..");
export const releasePlanPath = resolve(
  root,
  "packages/release/release-plan.json"
);
const generatedChangeset = resolve(root, ".changeset/rpc-plugins.md");
const markerPath = resolve(root, "packages/release/rpc-breaks");
const pluginDirectory = resolve(root, "integrations/pi");
const JsonObject = Schema.Record({
  key: Schema.String,
  value: JsonValueSchema,
});

export const readPreparedRelease = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem;
  return yield* fs
    .readFileString(releasePlanPath)
    .pipe(
      Effect.flatMap(
        Schema.decodeUnknown(Schema.parseJson(PreparedReleaseSchema))
      ),
      Effect.flatMap(validatePreparedRelease)
    );
});

export const verifyPublicBaseline = (plan: PreparedRelease) =>
  Effect.gen(function* () {
    const current = yield* readPublicBaseline;
    if (canonical(current) !== canonical(plan.baseline))
      return yield* new ReleasePreparationFailed({
        message:
          "The public release baseline changed; refresh the release plan before building",
      });
  });

const writePlan = (plan: PreparedRelease) =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    yield* validatePreparedRelease(plan);
    const contents = `${yield* Schema.encode(
      Schema.parseJson(PreparedReleaseSchema, { space: 2 })
    )(plan)}\n`;
    if (
      (yield* fs.exists(releasePlanPath)) &&
      (yield* fs.readFileString(releasePlanPath)) === contents
    )
      return;
    yield* fs.writeFileString(releasePlanPath, contents);
  });

export const prepareRelease = (mode: "detect" | "allocate" | "verify") =>
  Effect.scoped(
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem;
      const write = (path: string, contents: string) =>
        Effect.gen(function* () {
          if (yield* fs.exists(path))
            if ((yield* fs.readFileString(path)) === contents) return;
          yield* fs.writeFileString(path, contents);
        });
      const readPackage = (directory: string) =>
        fs
          .readFileString(`${directory}/package.json`)
          .pipe(
            Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(JsonObject)))
          );
      const cli = yield* readPackage(resolve(root, "packages/launcher"));
      const cliVersion = yield* Schema.decodeUnknown(Schema.String)(
        cli.version
      );
      // Prereleases prepare exactly like stable; only the plugin baseline tag differs.
      const channel = yield* readPrereleaseTag(root);
      // The committed plan is the only record of the daemon revision; it must already exist.
      if (!(yield* fs.exists(releasePlanPath)))
        return yield* new ReleasePreparationFailed({
          message:
            "packages/release/release-plan.json is missing; seed it with the current revision before preparing",
        });
      const existing = yield* readPreparedRelease;
      const baseline = yield* readPublicBaseline;

      // A merged allocation is not a new public baseline. Do not stack another
      // release PR on it while publication is outstanding. The same holds for a
      // plugin allocated for publication that npm does not serve yet.
      if (mode === "detect") {
        if (isAwaitingPublication(existing, cliVersion, baseline)) {
          yield* Console.log(
            `Release ${cliVersion} is awaiting publication; retaining its allocation.`
          );
          return { pending: true };
        }
        const unpublished = yield* Effect.forEach(
          existing.plugins.filter((plugin) => plugin.publish),
          ({ artifact }) =>
            publishedPluginIntegrity(artifact.name, artifact.version, root).pipe(
              Effect.map((integrity) =>
                integrity === null ? Option.some(artifact) : Option.none()
              )
            )
        ).pipe(Effect.map((results) => results.flatMap(Option.toArray)));
        if (unpublished.length > 0) {
          yield* Console.log(
            `${unpublished
              .map((artifact) => `${artifact.name}@${artifact.version}`)
              .join(", ")} awaiting publication; retaining the allocation.`
          );
          return { pending: true };
        }
      }
      const markers = (yield* fs.exists(markerPath))
        ? (yield* fs.readDirectory(markerPath))
            .filter((name) => name.endsWith(".json"))
            .sort()
        : [];
      const reasons = yield* Effect.forEach(markers, (name) =>
        fs.readFileString(`${markerPath}/${name}`).pipe(
          Effect.flatMap(
            Schema.decodeUnknown(
              Schema.parseJson(
                Schema.Struct({
                  reason: Schema.String.pipe(Schema.minLength(1)),
                })
              )
            )
          ),
          Effect.map((value) => value.reason)
        )
      );
      // Markers are consumed by allocation, so the plan's own breaks are never re-applied.
      const semanticBreaks = [...new Set(reasons)].sort();
      const fingerprint = yield* rpcFingerprint();
      const rpc = yield* allocateRpcVersion(existing, fingerprint, semanticBreaks);
      // Detect observes; it never advances the revision. Allocation advances it once per CLI version.
      const revision =
        mode === "detect"
          ? existing.revision
          : yield* allocateRevision(existing, cliVersion);
      const generation = <A>(run: () => Promise<A>) =>
        Effect.tryPromise({
          try: run,
          catch: (error) =>
            new ReleasePreparationFailed({
              message: `Version generation failed: ${String(error)}`,
            }),
        });
      if (mode === "verify") {
        if (
          canonical(existing.rpc) !== canonical(rpc) ||
          existing.cliVersion !== cliVersion ||
          existing.revision !== revision ||
          (yield* generation(readGeneratedRpcVersion)) !== rpc.version
        ) {
          return yield* new ReleasePreparationFailed({
            message:
              "RPC contract or release allocation is stale; run release preparation",
          });
        }
        yield* verifyPublicBaseline(existing);
      } else {
        // Generated identity must reflect this allocation before the plugin bundles it.
        yield* generation(() =>
          generateVersionFiles({ cliVersion, revision, rpcVersion: rpc.version })
        );
      }
      // Run in a new process: the generated RPC version must not come from this
      // process's already-evaluated protocol module cache.
      const buildCode = yield* Command.make("bun", "run", "build").pipe(
        Command.workingDirectory(pluginDirectory),
        Command.stdout("inherit"),
        Command.stderr("inherit"),
        Command.exitCode
      );
      if (buildCode !== 0)
        return yield* new ReleasePreparationFailed({
          message: "Plugin build failed",
        });
      const { metadata } = yield* verifyPluginContent(pluginDirectory);
      // The plugin's baseline is what npm serves, so a plugin-only publication is its own baseline.
      const previousPlugin = yield* publishedPlugin(
        "pi",
        metadata.name,
        root,
        channel
      );
      const candidate = yield* planPlugin(metadata, previousPlugin);
      if (mode === "detect") {
        const rpcChanged = canonical(existing.rpc) !== canonical(rpc);
        const declared = yield* declaredChangesetReleases(
          resolve(root, ".changeset"),
          generatedChangeset
        );
        const changeset = derivedChangeset({
          rpcChanged,
          releases: [
            ...(rpcChanged && !declared.has(CLI_PACKAGE_NAME)
              ? [CLI_PACKAGE_NAME]
              : []),
            ...(candidate.publish && !declared.has(metadata.name)
              ? [metadata.name]
              : []),
          ],
        });
        if (changeset !== undefined) yield* write(generatedChangeset, changeset);
        else if (yield* fs.exists(generatedChangeset))
          yield* fs.remove(generatedChangeset);
        yield* Console.log(
          changeset !== undefined
            ? "Prepared derived RPC/plugin changeset."
            : "RPC and shipped plugin changes are already declared."
        );
        return { pending: false };
      }
      if (mode === "verify") {
        const selected = existing.plugins.find(
          (plugin) => plugin.artifact.host === "pi"
        );
        if (selected === undefined)
          return yield* new ReleasePreparationFailed({
            message: "Release plan omitted the Pi plugin",
          });
        if (
          selected.artifact.name !== metadata.name ||
          selected.artifact.version !== metadata.version ||
          selected.artifact.rpcVersion !== metadata.rpcVersion ||
          selected.artifact.contentFingerprint !== metadata.contentFingerprint
        ) {
          return yield* new ReleasePreparationFailed({
            message: "Bundled plugin contents differ from the prepared release",
          });
        }
        yield* Console.log(
          "RPC contract, bundled plugin and exact CLI pins verified."
        );
        return { pending: false };
      }

      if ((yield* requireNpm(["--version"], root)).trim() !== "11.6.2")
        return yield* new ReleasePreparationFailed({
          message: "Release preparation requires npm 11.6.2",
        });
      const packageJson = yield* readPackage(pluginDirectory);
      let version = candidate.version;
      let artifact;
      if (!candidate.publish && Option.isSome(candidate.previous)) {
        artifact = candidate.previous.value;
        yield* write(
          `${pluginDirectory}/package.json`,
          `${JSON.stringify({ ...packageJson, version }, null, 2)}\n`
        );
      } else {
        const output = yield* fs.makeTempDirectoryScoped({
          prefix: "magnitude-plugin-allocation-",
        });
        // A version published by an interrupted prior release is immutable. Reuse
        // identical bytes or reserve the next patch before committing the plan.
        for (;;) {
          yield* write(
            `${pluginDirectory}/package.json`,
            `${JSON.stringify({ ...packageJson, version }, null, 2)}\n`
          );
          const code = yield* Command.make("bun", "run", "build").pipe(
            Command.workingDirectory(pluginDirectory),
            Command.stdout("inherit"),
            Command.stderr("inherit"),
            Command.exitCode
          );
          if (code !== 0)
            return yield* new ReleasePreparationFailed({
              message: "Plugin metadata rebuild failed",
            });
          const packed = yield* packPlugin(pluginDirectory, output);
          const published = yield* publishedPluginIntegrity(
            packed.name,
            packed.version,
            root
          );
          if (published === null || published === packed.integrity) {
            artifact = packed;
            break;
          }
          const next = inc(version, "patch");
          if (next === null)
            return yield* new ReleasePreparationFailed({
              message: "Cannot allocate the next plugin version",
            });
          version = next;
        }
      }
      const changelogPath = `${pluginDirectory}/CHANGELOG.md`;
      if (yield* fs.exists(changelogPath)) {
        const currentVersion = yield* Schema.decodeUnknown(Schema.String)(
          packageJson.version
        );
        yield* write(
          changelogPath,
          reconcilePluginChangelog(
            yield* fs.readFileString(changelogPath),
            currentVersion,
            version,
            candidate.publish
          )
        );
      }
      yield* writePlan({
        format: 2,
        baseline,
        cliVersion,
        revision,
        rpc,
        semanticBreaks,
        plugins: [{ artifact, publish: candidate.publish }],
      });
      for (const marker of markers) yield* fs.remove(`${markerPath}/${marker}`);
      yield* Console.log(
        `Prepared revision ${revision}, RPC ${rpc.version} and ${artifact.name}@${artifact.version}.`
      );
      return { pending: false };
    })
  );

if (import.meta.main) {
  const args = process.argv.slice(2);
  const mode = args.includes("--detect")
    ? "detect"
    : args.includes("--verify")
    ? "verify"
    : "allocate";
  BunRuntime.runMain(
    prepareRelease(mode).pipe(
      Effect.tap((result) =>
        Effect.gen(function* () {
          const output = process.env.GITHUB_OUTPUT;
          if (output)
            yield* (yield* FileSystem.FileSystem).writeFileString(
              output,
              `pending=${result.pending}\n`,
              { flag: "a" }
            );
        })
      ),
      Effect.provide(BunContext.layer)
    )
  );
}
