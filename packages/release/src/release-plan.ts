import { Effect, Option, Schema } from "effect";
import { gt, inc, prerelease } from "semver";
import {
  PluginArtifactSchema,
  PluginHostSchema,
  type PluginArtifact,
  type PluginHost,
  RpcReleaseSchema,
  type PluginContentManifest,
} from "./plugins";

/** The last stable CLI release. Plugins have their own baseline: what npm publishes as latest. */
export const PublicBaselineSchema = Schema.Struct({
  tag: Schema.String,
  sourceCommit: Schema.String.pipe(Schema.pattern(/^[a-f0-9]{40}$/)),
  cliVersion: Schema.String,
  rpc: Schema.optionalWith(RpcReleaseSchema, { as: "Option", exact: true }),
});
export type PublicBaseline = typeof PublicBaselineSchema.Type;
const Revision = Schema.Number.pipe(
  Schema.int(),
  Schema.positive(),
  Schema.lessThanOrEqualTo(Number.MAX_SAFE_INTEGER)
);

/** The one committed record of release identity: CLI version, daemon revision, RPC, plugins. */
export const PreparedReleaseSchema = Schema.Struct({
  format: Schema.Literal(2),
  baseline: Schema.optionalWith(PublicBaselineSchema, {
    as: "Option",
    exact: true,
  }),
  cliVersion: Schema.String,
  /** Private daemon coordination revision. Orders daemon processes; independent of the RPC version. */
  revision: Revision,
  rpc: RpcReleaseSchema,
  semanticBreaks: Schema.Array(Schema.String.pipe(Schema.minLength(1))),
  plugins: Schema.Array(
    Schema.Struct({
      artifact: PluginArtifactSchema,
      publish: Schema.Boolean,
    })
  ),
});
export type PreparedRelease = typeof PreparedReleaseSchema.Type;

export const isAwaitingPublication = (
  plan: PreparedRelease,
  cliVersion: string,
  baseline: Option.Option<PublicBaseline>
): boolean =>
  prerelease(plan.cliVersion) === null &&
  plan.cliVersion === cliVersion &&
  Option.getOrUndefined(
    Option.map(baseline, (baseline) => baseline.cliVersion)
  ) !== cliVersion;

/**
 * The revision advances by one whenever the CLI version changes since the previous plan,
 * prereleases included, so every distinct CLI version orders above the one before it.
 * The same CLI version always keeps its revision; preparation is idempotent.
 */
export const allocateRevision = (
  previous: PreparedRelease,
  cliVersion: string
) =>
  Effect.gen(function* () {
    if (previous.cliVersion === cliVersion) return previous.revision;
    if (previous.revision >= Number.MAX_SAFE_INTEGER)
      return yield* new ReleasePreparationFailed({
        message: "ACN revision space is exhausted",
      });
    return previous.revision + 1;
  });

/**
 * The RPC version is a monotonic counter over successive plans, in every channel: a changed
 * contract or a semantic-break marker takes the next number. A number never names two contracts.
 */
export const allocateRpcVersion = (
  previous: PreparedRelease,
  fingerprint: string,
  semanticBreaks: readonly string[]
) =>
  Effect.gen(function* () {
    const changed =
      previous.rpc.fingerprint !== fingerprint || semanticBreaks.length > 0;
    const version = previous.rpc.version + (changed ? 1 : 0);
    if (!Number.isSafeInteger(version))
      return yield* new ReleasePreparationFailed({
        message: "RPC version space is exhausted",
      });
    return { version, fingerprint };
  });

/** A human major/minor bump wins; a changed embedded SDK needs at least a patch. */
export const planPlugin = (
  metadata: PluginContentManifest,
  previous: Option.Option<PluginArtifact>
) =>
  Effect.gen(function* () {
    if (
      Option.isSome(previous) &&
      previous.value.rpcVersion === metadata.rpcVersion &&
      previous.value.contentFingerprint === metadata.contentFingerprint
    ) {
      return { publish: false, version: previous.value.version, previous };
    }
    const version = Option.match(previous, {
      onNone: () => metadata.version,
      onSome: (previous) =>
        gt(metadata.version, previous.version)
          ? metadata.version
          : inc(previous.version, "patch"),
    });
    if (version === null)
      return yield* new ReleasePreparationFailed({
        message: `Invalid plugin version ${metadata.version}`,
      });
    return { publish: true, version, previous };
  });

export class ReleasePreparationFailed extends Schema.TaggedError<ReleasePreparationFailed>()(
  "ReleasePreparationFailed",
  {
    message: Schema.String,
  }
) {}

export const validatePreparedRelease = (plan: PreparedRelease) =>
  Effect.gen(function* () {
    const prereleaseCli = prerelease(plan.cliVersion) !== null;
    const names = new Set<string>();
    const hosts = new Set<PluginHost>();
    for (const { artifact, publish } of plan.plugins) {
      // A prerelease CLI publishes prerelease plugins under its dist-tag; stable publishes stable.
      if (publish && (prerelease(artifact.version) !== null) !== prereleaseCli)
        return yield* new ReleasePreparationFailed({
          message: `Plugin ${artifact.name}@${artifact.version} cannot be published from CLI ${plan.cliVersion}; channels must match`,
        });
      if (
        artifact.rpcVersion !== plan.rpc.version ||
        names.has(artifact.name) ||
        hosts.has(artifact.host)
      ) {
        return yield* new ReleasePreparationFailed({
          message:
            "Plugin selection must be unique and match the CLI RPC version exactly",
        });
      }
      names.add(artifact.name);
      hosts.add(artifact.host);
    }
    const missingHosts = PluginHostSchema.literals.filter(
      (host) => !hosts.has(host)
    );
    if (missingHosts.length > 0)
      return yield* new ReleasePreparationFailed({
        message: `Missing plugin selection for hosts: ${missingHosts.join(
          ", "
        )}`,
      });
    return plan;
  });
