import { Atom, Result } from "@effect-atom/atom-react"
import type { AgentClient } from "../state/agent-client"
import { OnboardingModelSetup } from "./setup"

const makeSetupView = (client: AgentClient) => {
  const service = client.runtime.atom(OnboardingModelSetup)
  return Atom.make((get) => Result.flatMap(get(service), (setup) => get(setup.view)))
}

const setupViews = new WeakMap<AgentClient, ReturnType<typeof makeSetupView>>()

export const onboardingModelSetupViewAtom = (client: AgentClient) => {
  const existing = setupViews.get(client)
  if (existing !== undefined) return existing
  const view = makeSetupView(client)
  setupViews.set(client, view)
  return view
}
