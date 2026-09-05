import { Schema } from "effect";

export const Sha256Schema = Schema.String.pipe(
  Schema.pattern(/^[a-f0-9]{64}$/)
);
const Version = Schema.String.pipe(
  Schema.pattern(/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/)
);
const RpcVersion = Schema.Number.pipe(
  Schema.int(),
  Schema.positive(),
  Schema.lessThanOrEqualTo(Number.MAX_SAFE_INTEGER)
);
const PackageName = Schema.String.pipe(
  Schema.pattern(/^@magnitudedev\/[a-z0-9-]+$/)
);
const PackageFile = Schema.String.pipe(
  Schema.pattern(/^(?:dist\/[a-zA-Z0-9_./-]+|README\.md)$/),
  Schema.filter((path) => !path.split("/").includes(".."))
);

/** Generated manifest embedded in the plugin, describing its bundled content. */
export const PluginContentManifestSchema = Schema.Struct({
  name: PackageName,
  version: Version,
  rpcVersion: RpcVersion,
  contentFingerprint: Sha256Schema,
  files: Schema.Record({ key: PackageFile, value: Sha256Schema }),
});
export type PluginContentManifest = typeof PluginContentManifestSchema.Type;
export const PluginHostSchema = Schema.Literal("pi");
export type PluginHost = typeof PluginHostSchema.Type;
/** Exact immutable npm tarball selected by a CLI release. */
export const PluginArtifactSchema = Schema.Struct({
  host: PluginHostSchema,
  name: PackageName,
  version: Version,
  rpcVersion: RpcVersion,
  contentFingerprint: Sha256Schema,
  filename: Schema.String.pipe(Schema.pattern(/^[a-zA-Z0-9._-]+\.tgz$/)),
  integrity: Schema.String.pipe(Schema.pattern(/^sha512-[a-zA-Z0-9+/]+=*$/)),
});
export type PluginArtifact = typeof PluginArtifactSchema.Type;
export const RpcReleaseSchema = Schema.Struct({
  version: RpcVersion,
  fingerprint: Sha256Schema,
});

/** Validated npm installation fields; content fingerprinting excludes the assigned version. */
export const PluginPackageManifestSchema = Schema.Struct({
  name: PackageName,
  version: Version,
  type: Schema.Literal("module"),
  files: Schema.Array(Schema.String),
  pi: Schema.Struct({ extensions: Schema.Array(Schema.String) }),
  dependencies: Schema.Record({ key: Schema.String, value: Schema.String }),
  peerDependencies: Schema.Record({ key: Schema.String, value: Schema.String }),
});
export type PluginPackageManifest = typeof PluginPackageManifestSchema.Type;
