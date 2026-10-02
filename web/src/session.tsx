import { createContext, useContext, useMemo, type ReactNode } from "react"
import { Atom, useAtomSet } from "@effect-atom/atom-react"
import { Option } from "effect"
import type { ApplicationPage, ApplicationSession } from "@magnitudedev/client-common"

const SessionContext = createContext<ApplicationSession | null>(null)

export function SessionProvider({ session, children }: { session: ApplicationSession; children: ReactNode }) {
  return <SessionContext.Provider value={session}>{children}</SessionContext.Provider>
}

/** The connection's application session, resolved once by the app root before pages render. */
export const useSession = (): ApplicationSession => {
  const session = useContext(SessionContext)
  if (session === null) throw new Error("useSession is used outside SessionProvider")
  return session
}

export const useNavigate = () => {
  const session = useSession()
  return useAtomSet(useMemo(() => Atom.fn((page: ApplicationPage) => session.navigate(page)), [session]))
}

/** The OS of the machine running Magnitude, for commands the user runs there. */
export const useServerPlatform = (): string => {
  const session = useSession()
  return Option.getOrElse(session.serverPlatform, () => "linux")
}
