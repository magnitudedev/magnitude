import parseChangeset from "@changesets/parse";
import * as FileSystem from "@effect/platform/FileSystem";
import { Effect } from "effect";
import { CLI_PACKAGE_NAME } from "../src/contracts";
import { ReleasePreparationFailed } from "../src/release-plan";

export interface DerivedReleases {
  /** The RPC contract moved since the CLI baseline. */
  readonly rpcChanged: boolean;
  /** Packages whose shipped content changed and which no human changeset already names. */
  readonly releases: readonly string[];
}

/** Only what changed and is not already declared by a human changeset. Nothing else is generated. */
export const derivedChangeset = ({
  rpcChanged,
  releases,
}: DerivedReleases): string | undefined => {
  if (releases.length === 0) return undefined;
  const summary = rpcChanged
    ? "Update the RPC contract and matching bundled harness plugins."
    : "Rebuild bundled harness plugins against the current Magnitude SDK.";
  return `---\n${releases
    .map((name) => `${JSON.stringify(name)}: patch`)
    .join("\n")}\n---\n\n${summary}\n`;
};

/** Package names released by human changesets, excluding the generated file itself. */
export const declaredChangesetReleases = (
  directory: string,
  generated: string
) =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const names = new Set<string>();
    for (const name of yield* fs.readDirectory(directory)) {
      if (!name.endsWith(".md") || name === "README.md") continue;
      const path = `${directory}/${name}`;
      if (path === generated) continue;
      const source = yield* fs.readFileString(path);
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
