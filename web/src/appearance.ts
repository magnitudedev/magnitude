import { Atom } from "@effect-atom/atom-react"

/** Set at startup when the saved appearance couldn't be read; the window then follows the system. */
export const appearanceReadError = Atom.keepAlive(Atom.make<string | null>(null))
