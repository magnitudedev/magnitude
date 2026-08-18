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
import type { UpdateAction } from "@magnitudedev/release"
import type { AcnLifecycleState } from "@magnitudedev/sdk"
import { CliApp, type CliAppProps } from "../app"
import { AcnBootstrapScreen } from "../features/app-shell/acn-bootstrap"
import {
  UpdatePrompt,
  type UpdatePromptOutcome,
} from "../features/update/prompt"

export type CliRootState =
  | {
      readonly _tag: "UpdatePrompt"
      readonly currentVersion: string
      readonly latestVersion: string
      readonly action: UpdateAction
    }
  | {
      readonly _tag: "DaemonStartup"
      readonly lifecycle: AcnLifecycleState
    }
  | {
      readonly _tag: "Application"
      readonly platform: Platform
      readonly agentClient: AgentClientInstance
      readonly app: CliAppProps
    }

export const makeCliRootStateAtom = (
  initial: CliRootState,
): Atom.Writable<CliRootState> => Atom.make<CliRootState>(initial)

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
  onDaemonRetry,
  onDaemonQuit,
}: {
  readonly stateAtom: Atom.Atom<CliRootState>
  readonly onUpdateSelect: (outcome: UpdatePromptOutcome) => void
  readonly onDaemonRetry: () => void
  readonly onDaemonQuit: () => void
}): ReactNode {
  const state = useAtomValue(stateAtom)

  return (
    <box style={{ width: "100%", height: "100%" }}>
      {state._tag === "UpdatePrompt" ? (
        <UpdatePrompt
          currentVersion={state.currentVersion}
          latestVersion={state.latestVersion}
          action={state.action}
          onSelect={onUpdateSelect}
        />
      ) : state._tag === "DaemonStartup" ? (
        <AcnBootstrapScreen
          state={state.lifecycle}
          onRetry={onDaemonRetry}
          onQuit={onDaemonQuit}
        />
      ) : (
        <ApplicationRoot state={state} />
      )}
    </box>
  )
}
