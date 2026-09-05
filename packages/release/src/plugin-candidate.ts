import * as FileSystem from "@effect/platform/FileSystem";
import { Effect, Schema } from "effect";
import { canonical } from "@magnitudedev/utils/canonical-key";
import {
  PreparedReleaseSchema,
  validatePreparedRelease,
} from "./release-plan";
import { sha256 } from "./plugin-content";
import { verifyPluginArtifact, PluginArtifactError } from "./plugin-artifacts";

export const PluginAcceptanceReceiptSchema = Schema.Struct({
  planFingerprint: Schema.String,
  runtimes: Schema.Tuple(Schema.Literal("node"), Schema.Literal("bun")),
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
      verifyPluginArtifact(plugin.artifact, directory)
    );
    const receipt: PluginAcceptanceReceipt = {
      planFingerprint: sha256(canonical(plan)),
      runtimes: ["node", "bun"],
      artifacts: plan.plugins.map(({ artifact }) => ({
        filename: artifact.filename,
        integrity: artifact.integrity,
      })),
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
