import { Config, Effect, Option, Schema } from "effect";
import { RpcReleaseSchema } from "@magnitudedev/release/plugins";
import {
  ReleasePreparationFailed,
  type PublicBaseline,
} from "../src/release-plan";
import { isPrereleaseVersion } from "./release-channel";

const Releases = Schema.Array(
  Schema.Struct({
    draft: Schema.Boolean,
    prerelease: Schema.Boolean,
    tag_name: Schema.String,
    published_at: Schema.NullOr(Schema.String),
    assets: Schema.Array(
      Schema.Struct({ name: Schema.String, url: Schema.String })
    ),
  })
);
// Historical public manifests did not record an RPC allocation. Only baseline discovery
// accepts its absence; SDK admission and new releases require exact metadata.
const BaselineManifest = Schema.Struct({
  tag: Schema.String,
  version: Schema.String,
  sourceCommit: Schema.String.pipe(Schema.pattern(/^[a-f0-9]{40}$/)),
  rpc: Schema.optionalWith(RpcReleaseSchema, { exact: true, as: "Option" }),
});

export const readPublicBaseline = Effect.gen(function* () {
  const token = yield* Config.option(Config.string("GITHUB_TOKEN"));
  const repository = yield* Config.string("GITHUB_REPOSITORY").pipe(
    Config.withDefault("magnitudedev/magnitude")
  );
  const request = (url: string, accept = "application/vnd.github+json") =>
    Effect.tryPromise({
      try: async (signal) => {
        if (new URL(url).origin !== "https://api.github.com")
          throw new Error("Invalid GitHub release metadata origin");
        const response = await fetch(url, {
          headers: {
            accept,
            "x-github-api-version": "2022-11-28",
            ...Option.match(token, {
              onNone: () => ({}),
              onSome: (token) => ({ authorization: `Bearer ${token}` }),
            }),
          },
          signal,
        });
        if (!response.ok)
          throw new Error(
            `GitHub returned ${response.status} for release metadata`
          );
        return response.json() as Promise<unknown>;
      },
      catch: (error) =>
        new ReleasePreparationFailed({ message: String(error) }),
    }).pipe(Effect.timeout("30 seconds"));
  const releases: Array<(typeof Releases.Type)[number]> = [];
  for (let page = 1; ; page++) {
    const batch = yield* request(
      `https://api.github.com/repos/${repository}/releases?per_page=100&page=${page}`
    ).pipe(Effect.flatMap(Schema.decodeUnknown(Releases)));
    releases.push(...batch);
    if (batch.length < 100) break;
  }
  const latest = releases
    .filter(
      (
        release
      ): release is typeof release & { readonly published_at: string } =>
        !release.draft &&
        !release.prerelease &&
        release.published_at !== null &&
        release.tag_name.startsWith("@magnitudedev/cli@") &&
        !isPrereleaseVersion(
          release.tag_name.slice("@magnitudedev/cli@".length)
        )
    )
    .sort((a, b) => b.published_at.localeCompare(a.published_at))[0];
  if (latest === undefined) return Option.none<PublicBaseline>();
  const asset = latest.assets.find(
    (asset) => asset.name === "magnitude-release.json"
  );
  if (!asset)
    return yield* new ReleasePreparationFailed({
      message: `Public release ${latest.tag_name} has no manifest; refusing to guess its baseline`,
    });
  const manifest = yield* request(asset.url, "application/octet-stream").pipe(
    Effect.flatMap(Schema.decodeUnknown(BaselineManifest))
  );
  if (
    manifest.tag !== latest.tag_name ||
    manifest.tag !== `@magnitudedev/cli@${manifest.version}`
  )
    return yield* new ReleasePreparationFailed({
      message: "Public release manifest tag/version mismatch",
    });
  return Option.some<PublicBaseline>({
    tag: manifest.tag,
    cliVersion: manifest.version,
    sourceCommit: manifest.sourceCommit,
    rpc: manifest.rpc,
  });
});
