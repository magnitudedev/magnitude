import * as FileSystem from "@effect/platform/FileSystem";
import { createHash } from "node:crypto";
import { Effect, Schema } from "effect";
import { canonical } from "@magnitudedev/utils/canonical-key";
import { JsonValueSchema, type JsonValue } from "@magnitudedev/utils/schema";
import {
  PluginContentManifestSchema,
  PluginPackageManifestSchema,
  type PluginContentManifest,
} from "./plugins";

export const PLUGIN_METADATA_PATH = "dist/magnitude-plugin.json";
export const sha256 = (data: string | Uint8Array): string =>
  createHash("sha256").update(data).digest("hex");
export const packageContentFingerprint = (
  manifest: Readonly<Record<string, JsonValue>>,
  rpcVersion: number,
  files: Readonly<Record<string, string>>
): string => {
  // Keep every potentially install-relevant field, including fields added later.
  // Development tools and descriptive metadata do not change the installed API.
  const ignored = new Set([
    "version",
    "devDependencies",
    "scripts",
    "description",
    "keywords",
    "repository",
    "homepage",
    "bugs",
    "author",
    "contributors",
    "publishConfig",
  ]);
  const install = Object.fromEntries(
    Object.entries(manifest).filter(([key]) => !ignored.has(key))
  );
  if (
    manifest.scripts &&
    typeof manifest.scripts === "object" &&
    !Array.isArray(manifest.scripts)
  ) {
    const hooks = Object.fromEntries(
      Object.entries(manifest.scripts).filter(([key]) =>
        ["preinstall", "install", "postinstall"].includes(key)
      )
    );
    if (Object.keys(hooks).length > 0) install.scripts = hooks;
  }
  return sha256(canonical({ install, rpcVersion, files }));
};

export const inspectPluginContent = (directory: string, rpcVersion: number) =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const raw = yield* fs
      .readFileString(`${directory}/package.json`)
      .pipe(
        Effect.flatMap(
          Schema.decodeUnknown(
            Schema.parseJson(
              Schema.Record({ key: Schema.String, value: JsonValueSchema })
            )
          )
        )
      );
    const manifest = yield* Schema.decodeUnknown(PluginPackageManifestSchema)(raw);
    const optional = yield* Schema.decodeUnknown(
      Schema.Record({ key: Schema.String, value: Schema.String })
    )(raw.optionalDependencies ?? {});
    for (const [name, version] of Object.entries({
      ...manifest.dependencies,
      ...optional,
      ...manifest.peerDependencies,
    })) {
      if (
        name.startsWith("@magnitudedev/") ||
        /^(workspace:|file:|link:)/.test(version)
      )
        return yield* Effect.dieMessage(
          `Private runtime dependency escaped the plugin: ${name}`
        );
    }
    const paths: string[] = ["README.md"];
    const visit = (
      path: string
    ): Effect.Effect<void, import("@effect/platform/Error").PlatformError> =>
      Effect.gen(function* () {
        for (const name of (yield* fs.readDirectory(
          `${directory}/${path}`
        )).sort()) {
          const child = `${path}/${name}`;
          if (child === PLUGIN_METADATA_PATH) continue;
          const stat = yield* fs.stat(`${directory}/${child}`);
          if (stat.type === "Directory") yield* visit(child);
          else if (stat.type === "File") paths.push(child);
          else
            return yield* Effect.dieMessage(
              `Unsupported plugin file: ${child}`
            );
        }
      });
    yield* visit("dist");
    const files = Object.fromEntries(
      yield* Effect.forEach(paths.sort(), (path) =>
        fs
          .readFile(`${directory}/${path}`)
          .pipe(Effect.map((bytes) => [path, sha256(bytes)] as const))
      )
    );
    const metadata: PluginContentManifest = {
      name: manifest.name,
      version: manifest.version,
      rpcVersion,
      files,
      contentFingerprint: packageContentFingerprint(raw, rpcVersion, files),
    };
    return { manifest, metadata };
  });

export const verifyPluginContent = (directory: string) =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const recorded = yield* fs
      .readFileString(`${directory}/${PLUGIN_METADATA_PATH}`)
      .pipe(
        Effect.flatMap(
          Schema.decodeUnknown(Schema.parseJson(PluginContentManifestSchema))
        )
      );
    const actual = yield* inspectPluginContent(directory, recorded.rpcVersion);
    if (canonical(recorded) !== canonical(actual.metadata))
      return yield* Effect.fail(new PluginContentMismatch({ directory }));
    return actual;
  });
export class PluginContentMismatch extends Schema.TaggedError<PluginContentMismatch>()(
  "PluginContentMismatch",
  { directory: Schema.String }
) {}
