import type { HarnessId } from "@magnitudedev/client-common"
import type { HarnessConnector } from "./contract"
import { makeClaudeCodeConnector } from "./connectors/claude-code"
import { makeClineConnector } from "./connectors/cline"
import {
  makeCodexConnector,
  type CodexBundledCatalogReader,
} from "./connectors/codex"
import { makeHermesConnector } from "./connectors/hermes"
import { makeMagnitudeConnector } from "./connectors/magnitude"
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
  readonly readCodexBundledCatalog?: CodexBundledCatalogReader
}

export const makeHarnessConnectorRegistry = (
  paths: HarnessConnectionPaths,
  options: HarnessConnectorRegistryOptions = {},
): HarnessConnectorRegistry => {
  const ordered = [
    makeMagnitudeConnector(),
    makePiConnector(paths),
    makeOpenCodeConnector(paths),
    makeHermesConnector(paths),
    makeOpenClawConnector(paths),
    makeCodexConnector(paths, options.readCodexBundledCatalog),
    makeClaudeCodeConnector(paths),
    makeOhMyPiConnector(paths),
    makeClineConnector(paths),
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
