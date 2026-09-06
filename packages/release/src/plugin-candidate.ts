import * as FileSystem from "@effect/platform/FileSystem";
import { Effect, Schema } from "effect";
import { canonical } from "@magnitudedev/utils/canonical-key";
import {
  PreparedReleaseSchema,
  validatePreparedRelease,
} from "./release-plan";
import { sha256 } from "./plugin-content";
import { artifactIntegrity, verifyPluginArtifact, PluginArtifactError } from "./plugin-artifacts";
import { verifyHermesPluginArtifact } from "./hermes-plugin-artifact";

export const PluginAcceptanceReceiptSchema = Schema.Struct({
  planFingerprint: Schema.String,
  runtimes: Schema.Tuple(Schema.Literal("node"), Schema.Literal("bun"), Schema.Literal("hermes")),
  artifacts: Schema.Array(
    Schema.Struct({ filename: Schema.String, integrity: Schema.String })
  ),
});
export type PluginAcceptanceReceipt = typeof PluginAcceptanceReceiptSchema.Type;

export const readPluginCandidate = (directory: string) =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const plan = yield* fs
      .readFileString(`${directory}/release-plan.json`)
      .pipe(
        Effect.flatMap(
          Schema.decodeUnknown(Schema.parseJson(PreparedReleaseSchema))
        ),
        Effect.flatMap(validatePreparedRelease)
      );
    const paths = yield* Effect.forEach(plan.plugins, (plugin) =>
      plugin.artifact.host === "hermes" ? verifyHermesPluginArtifact(plugin.artifact, directory) : verifyPluginArtifact(plugin.artifact, directory)
    );
    const receipt: PluginAcceptanceReceipt = {
      planFingerprint: sha256(canonical(plan)),
      runtimes: ["node", "bun", "hermes"],
      artifacts: yield* Effect.forEach(plan.plugins, ({ artifact }) => fs.readFile(`${directory}/${artifact.filename}`).pipe(
        Effect.map(bytes => ({ filename: artifact.filename, integrity: artifactIntegrity(bytes) })),
      )),
    };
    return { plan, paths, receipt };
  });

export const readAcceptedPluginCandidate = (directory: string) =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const candidate = yield* readPluginCandidate(directory);
    const receipt = yield* fs
      .readFileString(`${directory}/accepted.json`)
      .pipe(
        Effect.flatMap(
          Schema.decodeUnknown(Schema.parseJson(PluginAcceptanceReceiptSchema))
        )
      );
    if (canonical(receipt) !== canonical(candidate.receipt))
      return yield* new PluginArtifactError({
        message:
          "Plugin acceptance does not cover this exact release plan and tarballs",
      });
    return candidate;
  });
