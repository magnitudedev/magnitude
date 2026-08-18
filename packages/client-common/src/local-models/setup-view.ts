import { Atom, Result } from "@effect-atom/atom-react"
import type { AgentClientInstance } from "../state/agent-client"
import { OnboardingModelSetup } from "./setup"

const makeSetupView = (client: AgentClientInstance) => {
  const service = client.effectQuery.runtime.atom(OnboardingModelSetup)
  return Atom.make((get) => Result.flatMap(get(service), (setup) => get(setup.view)))
}

const setupViews = new WeakMap<AgentClientInstance, ReturnType<typeof makeSetupView>>()

export const onboardingModelSetupViewAtom = (client: AgentClientInstance) => {
  const existing = setupViews.get(client)
  if (existing !== undefined) return existing
  const view = makeSetupView(client)
  setupViews.set(client, view)
  return view
}
