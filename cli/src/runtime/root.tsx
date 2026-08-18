import type { ReactNode } from "react"
import { Atom, useAtomValue } from "@effect-atom/atom-react"
import type {
  AgentClientInstance,
  Platform,
} from "@magnitudedev/client-common"
import {
  AgentClientProvider,
  DisplayViewControllerProvider,
  PlatformProvider,
} from "@magnitudedev/client-common"
import type { UpdateAction } from "@magnitudedev/sdk"
import { CliApp, type CliAppProps } from "../app"
import {
  UpdatePrompt,
  type UpdatePromptOutcome,
} from "../features/update/prompt"
import { useTheme } from "../hooks/use-theme"

export type CliStartupStage = "UpdateCheck" | "Platform" | "ClientPreflight"

export type CliRootState =
  | {
      readonly _tag: "Starting"
      readonly stage: CliStartupStage
    }
  | {
      readonly _tag: "UpdateAvailable"
      readonly currentVersion: string
      readonly latestVersion: string
      readonly action: UpdateAction
    }
  | {
      readonly _tag: "Application"
      readonly platform: Platform
      readonly agentClient: AgentClientInstance
      readonly app: CliAppProps
    }

export const makeCliRootStateAtom = (): Atom.Writable<CliRootState> =>
  Atom.make<CliRootState>({ _tag: "Starting", stage: "UpdateCheck" })

function StartingScreen({ stage }: { readonly stage: CliStartupStage }): ReactNode {
  const theme = useTheme()
  const detail = stage === "Platform"
    ? "Starting local services…"
    : stage === "ClientPreflight"
      ? "Preparing your workspace…"
      : "Starting Magnitude…"

  return (
    <box
      style={{
        width: "100%",
        height: "100%",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <text style={{ fg: theme.text.supporting }}>{detail}</text>
    </box>
  )
}

function ApplicationRoot({
  state,
}: {
  readonly state: Extract<CliRootState, { readonly _tag: "Application" }>
}): ReactNode {
  return (
    <PlatformProvider platform={state.platform}>
      <AgentClientProvider tag={state.agentClient}>
        <DisplayViewControllerProvider>
          <CliApp {...state.app} />
        </DisplayViewControllerProvider>
      </AgentClientProvider>
    </PlatformProvider>
  )
}

export function CliStartupRoot({
  stateAtom,
  onUpdateSelect,
}: {
  readonly stateAtom: Atom.Atom<CliRootState>
  readonly onUpdateSelect: (outcome: UpdatePromptOutcome) => void
}): ReactNode {
  const state = useAtomValue(stateAtom)

  return (
    <box style={{ width: "100%", height: "100%" }}>
      {state._tag === "Starting" ? (
        <StartingScreen stage={state.stage} />
      ) : state._tag === "UpdateAvailable" ? (
        <UpdatePrompt
          currentVersion={state.currentVersion}
          latestVersion={state.latestVersion}
          action={state.action}
          onSelect={onUpdateSelect}
        />
      ) : (
        <ApplicationRoot state={state} />
      )}
    </box>
  )
}
