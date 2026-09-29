import { Option } from "effect"

/**
 * Under `electron-vite dev` the renderer comes from a Vite server owned by the parent process,
 * and that process exits together with Electron. `app.relaunch` would inherit the dead server's
 * URL and open an empty window, so a development relaunch is instead signalled to the
 * supervising dev script through Electron's exit code.
 */
export interface DevelopmentRelaunch {
  readonly showWindow: boolean
}

const exitCodes = { showWindow: 86, background: 87 } as const

export const rendererServedByDevServer = (environment: Readonly<Record<string, string | undefined>>): boolean =>
  environment.ELECTRON_RENDERER_URL !== undefined

export const developmentRelaunchExitCode = (relaunch: DevelopmentRelaunch): number =>
  relaunch.showWindow ? exitCodes.showWindow : exitCodes.background

export const developmentRelaunchFromExitCode = (code: number | null): Option.Option<DevelopmentRelaunch> =>
  code === exitCodes.showWindow ? Option.some({ showWindow: true })
    : code === exitCodes.background ? Option.some({ showWindow: false })
    : Option.none()
