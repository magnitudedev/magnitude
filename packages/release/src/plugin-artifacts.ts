import * as Command from "@effect/platform/Command";
import * as FileSystem from "@effect/platform/FileSystem";
import { createHash } from "node:crypto";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { createGunzip } from "node:zlib";
import { Duration, Effect, Either, Option, Schema, Stream } from "effect";
import { extract } from "tar-stream";
import { PLUGIN_METADATA_PATH, verifyPluginContent } from "./plugin-content";
import {
  PluginArtifactSchema,
  PluginContentManifestSchema,
  type PluginArtifact,
  type PluginHost,
} from "./plugins";

export class PluginArtifactError extends Schema.TaggedError<PluginArtifactError>()(
  "PluginArtifactError",
  { message: Schema.String }
) {}

export const npm = (args: readonly string[], cwd: string) =>
  Effect.scoped(
    Effect.gen(function* () {
      const child = yield* Command.make("npm", ...args).pipe(
        Command.workingDirectory(cwd),
        Command.start
      );
      const text = <E, R>(stream: Stream.Stream<Uint8Array, E, R>) =>
        stream.pipe(
          Stream.decodeText(),
          Stream.runFold("", (text, chunk) =>
            (text + chunk).slice(-1024 * 1024)
          )
        );
      const [code, stdout, stderr] = yield* Effect.all(
        [child.exitCode, text(child.stdout), text(child.stderr)],
        { concurrency: "unbounded" }
      );
      return { code: Number(code), stdout, stderr };
    })
  ).pipe(Effect.timeout("5 minutes"));

export const requireNpm = (args: readonly string[], cwd: string) =>
  npm(args, cwd).pipe(
    Effect.flatMap((result) =>
      result.code === 0
        ? Effect.succeed(result.stdout)
        : Effect.fail(
            new PluginArtifactError({
              message: `npm ${args[0]} failed: ${result.stderr}`,
            })
          )
    )
  );

export const artifactIntegrity = (bytes: Uint8Array): string =>
  `sha512-${createHash("sha512").update(bytes).digest("base64")}`;
const PackResult = Schema.Array(
  Schema.Struct({
    filename: Schema.String.pipe(Schema.pattern(/^[a-zA-Z0-9._-]+\.tgz$/)),
    integrity: Schema.String,
    files: Schema.Array(Schema.Struct({ path: Schema.String })),
  })
);

export const packPlugin = (directory: string, output: string) =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const { metadata } = yield* verifyPluginContent(directory);
    yield* fs.makeDirectory(output, { recursive: true });
    const packed = yield* requireNpm(
      ["pack", "--ignore-scripts", "--json", "--pack-destination", output],
      directory
    ).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(PackResult))));
    if (packed.length !== 1)
      return yield* new PluginArtifactError({
        message: "Expected exactly one plugin tarball",
      });
    const artifact = packed[0]!;
    const expected = [
      ...Object.keys(metadata.files),
      "package.json",
      "dist/magnitude-plugin.json",
    ].sort();
    if (
      JSON.stringify(artifact.files.map((file) => file.path).sort()) !==
      JSON.stringify(expected)
    )
      return yield* new PluginArtifactError({
        message: "npm packed files outside the plugin content manifest",
      });
    const integrity = artifactIntegrity(
      yield* fs.readFile(`${output}/${artifact.filename}`)
    );
    if (integrity !== artifact.integrity)
      return yield* new PluginArtifactError({
        message: "npm pack integrity mismatch",
      });
    return yield* Schema.decodeUnknown(PluginArtifactSchema)({
      host: "pi",
      name: metadata.name,
      version: metadata.version,
      rpcVersion: metadata.rpcVersion,
      contentFingerprint: metadata.contentFingerprint,
      filename: artifact.filename,
      integrity,
    });
  });

export const verifyPluginArtifact = (
  artifact: PluginArtifact,
  directory: string
) =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const path = `${directory}/${artifact.filename}`;
    if (artifactIntegrity(yield* fs.readFile(path)) !== artifact.integrity)
      return yield* new PluginArtifactError({
        message: `Tarball integrity mismatch: ${artifact.filename}`,
      });
    return path;
  });

export const publishedPluginIntegrity = (
  name: string,
  version: string,
  cwd: string
) =>
  Effect.gen(function* () {
    const result = yield* npm(
      ["view", `${name}@${version}`, "dist.integrity", "--json"],
      cwd
    );
    if (result.code === 0)
      return yield* Schema.decodeUnknown(Schema.parseJson(Schema.String))(
        result.stdout
      );
    if (result.stderr.includes("E404")) return null;
    return yield* new PluginArtifactError({
      message: `Cannot determine whether ${name}@${version} is published: ${result.stderr}`,
    });
  });

/** npm's tarball naming for a scoped package: `@scope/name` becomes `scope-name-<version>.tgz`. */
export const pluginTarballFilename = (name: string, version: string): string =>
  `${name.replace(/^@/, "").replace("/", "-")}-${version}.tgz`;

/** Read one entry of a gzipped tarball held in memory. */
export const readTarballEntry = (bytes: Uint8Array, entry: string) =>
  Effect.async<Uint8Array, PluginArtifactError>((resume) => {
    const reader = extract();
    let found: Buffer[] | undefined;
    reader.on("entry", (header, stream, next) => {
      if (header.name !== entry) {
        stream.resume();
        stream.on("end", next);
        return;
      }
      const chunks: Buffer[] = [];
      stream.on("data", (chunk: Buffer) => chunks.push(chunk));
      stream.on("end", () => {
        found = chunks;
        next();
      });
    });
    pipeline(Readable.from([Buffer.from(bytes)]), createGunzip(), reader).then(
      () =>
        resume(
          found === undefined
            ? Effect.fail(
                new PluginArtifactError({
                  message: `Tarball has no ${entry} entry`,
                })
              )
            : Effect.succeed(new Uint8Array(Buffer.concat(found)))
        ),
      (cause) =>
        resume(
          Effect.fail(
            new PluginArtifactError({
              message: `Cannot read tarball: ${String(cause)}`,
            })
          )
        )
    );
    return Effect.sync(() => reader.destroy());
  });

const PublishedVersion = Schema.Struct({
  version: Schema.String,
  "dist.tarball": Schema.String,
  "dist.integrity": Schema.String,
});

/**
 * The plugin's public baseline: the version npm serves under the release channel's dist-tag,
 * falling back to `latest`, described by the content manifest inside its tarball.
 */
export const publishedPlugin = (
  host: PluginHost,
  name: string,
  cwd: string,
  channel: Option.Option<string> = Option.none()
) =>
  Effect.gen(function* () {
    const view = (tag: string) =>
      Effect.gen(function* () {
        const result = yield* npm(
          ["view", `${name}@${tag}`, "version", "dist.tarball", "dist.integrity", "--json"],
          cwd
        );
        if (result.code === 0) return Option.some(result.stdout);
        if (result.stderr.includes("E404")) return Option.none<string>();
        return yield* new PluginArtifactError({
          message: `Cannot read the published ${name}@${tag}: ${result.stderr}`,
        });
      });
    const viewed = yield* Option.match(channel, {
      onNone: () => view("latest"),
      onSome: (tag) =>
        view(tag).pipe(
          Effect.flatMap((found) =>
            Option.isSome(found) ? Effect.succeed(found) : view("latest")
          )
        ),
    });
    if (Option.isNone(viewed)) return Option.none<PluginArtifact>();
    const published = yield* Schema.decodeUnknown(
      Schema.parseJson(PublishedVersion)
    )(viewed.value).pipe(
      Effect.mapError(
        () =>
          new PluginArtifactError({
            message: `npm view returned an unexpected shape for ${name}`,
          })
      )
    );
    const bytes = yield* Effect.tryPromise({
      try: async (signal) => {
        const response = await fetch(published["dist.tarball"], { signal });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return new Uint8Array(await response.arrayBuffer());
      },
      catch: (error) =>
        new PluginArtifactError({
          message: `Cannot download the published ${name}: ${String(error)}`,
        }),
    }).pipe(Effect.timeout("2 minutes"));
    const integrity = artifactIntegrity(bytes);
    if (integrity !== published["dist.integrity"])
      return yield* new PluginArtifactError({
        message: `Published ${name}@${published.version} does not match its registry integrity`,
      });
    const manifest = yield* readTarballEntry(
      bytes,
      `package/${PLUGIN_METADATA_PATH}`
    ).pipe(
      Effect.flatMap((entry) =>
        Schema.decodeUnknown(Schema.parseJson(PluginContentManifestSchema))(
          new TextDecoder().decode(entry)
        )
      ),
      Effect.mapError(
        (error) =>
          new PluginArtifactError({
            message: `Published ${name}@${published.version} has no valid content manifest: ${String(error)}`,
          })
      )
    );
    if (manifest.name !== name || manifest.version !== published.version)
      return yield* new PluginArtifactError({
        message: `Published ${name}@${published.version} content manifest names a different package`,
      });
    return Option.some<PluginArtifact>({
      host,
      name,
      version: published.version,
      rpcVersion: manifest.rpcVersion,
      contentFingerprint: manifest.contentFingerprint,
      filename: pluginTarballFilename(name, published.version),
      integrity,
    });
  });

export interface RegistryReadBack {
  readonly attempts: number;
  readonly interval: Duration.DurationInput;
}
/** npm's registry lags a fresh publish; sixty seconds of polling covers it. */
export const REGISTRY_READ_BACK: RegistryReadBack = {
  attempts: 20,
  interval: "3 seconds",
};

/**
 * Wait for the registry to expose exactly the bytes that were published. A missing
 * version or a read failure is retried; a different integrity is final, because a
 * registry version is immutable. Failures name what was observed.
 */
export const awaitIntegrity = <E, R>(
  read: Effect.Effect<string | null, E, R>,
  expected: string,
  policy: RegistryReadBack = REGISTRY_READ_BACK
) =>
  Effect.gen(function* () {
    let last = "the registry did not expose the version";
    for (let attempt = 1; attempt <= policy.attempts; attempt++) {
      const observed = yield* Effect.either(read);
      if (Either.isRight(observed)) {
        if (observed.right === expected) return;
        if (observed.right !== null)
          return yield* new PluginArtifactError({
            message: `Registry integrity ${observed.right} differs from the accepted ${expected}; never overwrite a version`,
          });
      } else last = String(observed.left);
      if (attempt < policy.attempts) yield* Effect.sleep(policy.interval);
    }
    return yield* new PluginArtifactError({
      message: `Registry did not expose the accepted integrity ${expected} after ${policy.attempts} attempts: ${last}`,
    });
  });

export const awaitPublishedIntegrity = (
  name: string,
  version: string,
  expected: string,
  cwd: string
) => awaitIntegrity(publishedPluginIntegrity(name, version, cwd), expected);

export const verifyPublishedPlugins = (
  artifacts: readonly PluginArtifact[],
  cwd: string
) =>
  Effect.forEach(
    artifacts,
    (artifact) =>
      publishedPluginIntegrity(artifact.name, artifact.version, cwd).pipe(
        Effect.flatMap((integrity) =>
          integrity === artifact.integrity
            ? Effect.void
            : Effect.fail(
                new PluginArtifactError({
                  message: `Required plugin ${artifact.name}@${artifact.version} is not published with the selected integrity`,
                })
              )
        )
      ),
    { discard: true }
  );
