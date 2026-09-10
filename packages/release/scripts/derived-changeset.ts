import parseChangeset from "@changesets/parse";
import * as FileSystem from "@effect/platform/FileSystem";
import { Effect } from "effect";
import { CLI_PACKAGE_NAME } from "../src/contracts";
import { ReleasePreparationFailed } from "../src/release-plan";

export interface DerivedReleases {
  /** The RPC version the changed contract allocates. */
  readonly rpcVersion: number;
  /** Packages that must follow the new contract and which no human changeset already names. */
  readonly releases: readonly string[];
}

/**
 * A derived changeset exists for one reason: the RPC contract changed, so the CLI and every
 * bundled plugin must ship against the new version. Nothing else is ever derived; a plugin
 * change without a human changeset does not ship.
 */
export const derivedChangeset = ({
  rpcVersion,
  releases,
}: DerivedReleases): string | undefined => {
  if (releases.length === 0) return undefined;
  return `---\n${releases
    .map((name) => `${JSON.stringify(name)}: patch`)
    .join("\n")}\n---\n\nUpdate the RPC contract to v${rpcVersion} and rebuild the bundled harness plugins against it.\n`;
};

/** One file per RPC version: reusing a name would make changelog attribution find an older commit. */
export const derivedChangesetName = (rpcVersion: number) => `rpc-v${rpcVersion}.md`;
export const isDerivedChangeset = (name: string) => /^rpc-v\d+\.md$/.test(name);

/** Package names released by human changesets, excluding derived files. */
export const declaredChangesetReleases = (directory: string) =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const names = new Set<string>();
    for (const name of yield* fs.readDirectory(directory)) {
      if (!name.endsWith(".md") || name === "README.md" || isDerivedChangeset(name)) continue;
      const source = yield* fs.readFileString(`${directory}/${name}`);
      const parsed = yield* Effect.try({
        try: () => parseChangeset(source),
        catch: (error) =>
          new ReleasePreparationFailed({
            message: `Cannot parse changeset ${name}: ${String(error)}`,
          }),
      });
      for (const release of parsed.releases) names.add(release.name);
    }
    return names;
  });

export { CLI_PACKAGE_NAME };
