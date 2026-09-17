import { harnessEndpoints } from "./shared"
import type { HarnessId } from "@magnitudedev/client-common"
import type { HarnessCompanionPackage, HarnessConnector } from "./contract"
import { makeClaudeCodeConnector } from "./connectors/claude-code"
import { makeClineConnector } from "./connectors/cline"
import {
  makeCodexConnector,
  type CodexBundledCatalogReader,
} from "./connectors/codex"
import { makeHermesConnector } from "./connectors/hermes"
import { makeOhMyPiConnector } from "./connectors/oh-my-pi"
import { makeOpenClawConnector } from "./connectors/openclaw"
import { makeOpenCodeConnector } from "./connectors/opencode"
import { makePiConnector } from "./connectors/pi"
import type { HarnessConnectionPaths } from "./paths"

export interface HarnessConnectorRegistry {
  readonly ordered: ReadonlyArray<HarnessConnector>
  readonly get: (harness: HarnessId) => HarnessConnector
}

export interface HarnessConnectorRegistryOptions {
  readonly serviceEndpoint?: string
  readonly readCodexBundledCatalog?: CodexBundledCatalogReader
  /** Package lifecycle injection used by deterministic connector tests. */
  readonly piCompanion?: HarnessCompanionPackage
  /** Alternate Pi package source used by the repository's local development launcher. */
  readonly piCompanionSource?: string
}

export const makeHarnessConnectorRegistry = (
  paths: HarnessConnectionPaths,
  options: HarnessConnectorRegistryOptions = {},
): HarnessConnectorRegistry => {
  const endpoints = harnessEndpoints(options.serviceEndpoint)
  const ordered = [
    makePiConnector(paths, {
      baseUrl: endpoints.openai,
      ...(options.piCompanion === undefined ? {} : { companion: options.piCompanion }),
      ...(options.piCompanionSource === undefined ? {} : { packageSource: options.piCompanionSource }),
    }),
    makeOpenCodeConnector(paths, endpoints.openai),
    makeHermesConnector(paths, endpoints.openai),
    makeOpenClawConnector(paths, endpoints.openai),
    makeCodexConnector(paths, options.readCodexBundledCatalog, endpoints.codex),
    makeClaudeCodeConnector(paths, endpoints.anthropic),
    makeOhMyPiConnector(paths, endpoints.openai),
    makeClineConnector(paths, endpoints.openai),
  ]
  return {
    ordered,
    get: (harness) => {
      const connector = ordered.find((candidate) => candidate.id === harness)
      if (connector === undefined) throw new Error(`No harness connector is registered for ${harness}`)
      return connector
    },
  }
}
