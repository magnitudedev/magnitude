import { Effect, Option } from "effect"
import { publishedPluginIntegrity, PluginArtifactError } from "./plugin-artifacts"
import { publishedHermesRevision } from "./hermes-plugin-artifact"
import type { PluginArtifact } from "./plugins"

/** Every CLI pin must be publicly retrievable before that CLI is published. */
export const verifyPublishedPlugins = (artifacts: readonly PluginArtifact[], cwd: string) => Effect.forEach(
  artifacts,
  artifact => Effect.gen(function* () {
    const matches = artifact.host === "pi"
      ? (yield* publishedPluginIntegrity(artifact.name, artifact.version, cwd)) === artifact.integrity
      : Option.contains(artifact.revision)(yield* publishedHermesRevision(artifact, cwd))
    if (!matches) return yield* new PluginArtifactError({
      message: `Required plugin ${artifact.name}@${artifact.version} is not published with the selected identity`,
    })
  }),
  { discard: true },
)
