import { resolve } from 'path'
import { createCliRenderer } from '@opentui/core'
import { createRoot } from '@opentui/react'
import { Command } from '@commander-js/extra-typings'
import { Atom, RegistryProvider } from '@effect-atom/atom-react'
import {
  createAgentClient,
  AgentClientProvider,
  PlatformProvider,
  DisplayViewControllerProvider,
  stopDisplayViewController,
} from '@magnitudedev/client-common'
import { CliApp, type SessionStart } from './app'
import type { AuthSource } from './state/cli-atoms'
import { getLastSessionId } from './state/last-session'
import { CLI_VERSION } from './version'
import { installGracefulShutdownHandlers } from './utils/graceful-shutdown'
import { createTerminalPlatform } from './platform/terminal'
import { makeCliEffectLoggingLayer } from './platform/effect-logger'

/** One-time env-var auth resolution (spec §2.9) — not reactive. */
function resolveEnvAuth(): AuthSource {
  const envKey = process.env.MAGNITUDE_API_KEY
  if (envKey && envKey.trim()) {
    return { source: 'env', key: envKey, envVarName: 'MAGNITUDE_API_KEY' }
  }
  return { source: 'none' }
}

async function main() {
  const program = new Command()
    .name('magnitude')
    .version(CLI_VERSION)
    .option('--resume [id]', 'Resume the most recent chat session or a specific session by ID')
    .option('--debug', 'Enable debug mode with debug panel')
    .option('--autopilot', 'Launch with autopilot enabled')
    .option('--prompt <text>', 'Start session with an initial user message')
    .option('--headless', 'Run in headless mode (no TUI, output to stdout)')
    .option('--disable-shell-safeguards', 'Disable shell command classification safeguards')
    .option('--disable-cwd-safeguards', 'Disable working directory boundary safeguards')
    .option('--atif <path>', 'Write ATIF trajectory to the specified path')
    .option('--goal <objective>', 'Start a goal for the session')
    .option('--solo', 'Run without worker/task tools')
    .option('--system-override <text>', 'Override leader system prompt with raw text')
    .option('--setup', 'Rerun Local Models and Cloud Fallback setup')

    .action(async (opts) => {

      const sessionStart: SessionStart = opts.resume === undefined
        ? { _tag: 'new' }
        : opts.resume === true
          ? { _tag: 'latest' }
          : { _tag: 'resume', sessionId: opts.resume }

      // Headless mode is temporarily disabled while the CLI transitions to a
      // pure SDK/RPC client architecture. It needs a daemon-backed persistence
      // design before it can run again.
      if (opts.headless) {
        process.stderr.write('Error: --headless is temporarily disabled. Use the TUI mode.\n')
        process.exit(1)
      }

      const isDev = import.meta.url.endsWith('.tsx') || (process.argv[1]?.endsWith('.tsx') ?? false)
      const acnSourcePath = resolve(import.meta.dir, '..', '..', 'packages', 'acn', 'src', 'binary.ts')
      const spawnCommand = isDev
        ? ['bun', acnSourcePath, 'serve', '--register', ...(opts.debug ? ['--debug'] : [])]
        : undefined

      const effectLoggingLayer = makeCliEffectLoggingLayer({ debug: opts.debug === true })
      Atom.runtime.addGlobalLayer(effectLoggingLayer)
      const platform = createTerminalPlatform({ spawnCommand, debug: opts.debug, effectLoggingLayer })
      const agentClientTag = createAgentClient(platform.protocolLayer)
      const renderer = await createCliRenderer({
        exitOnCtrlC: false, // We handle Ctrl+C manually for two-tap exit
      })

      // Terminal background detection is handled by useTerminalBgDetection
      // inside the React tree (needs atom registry to write to themeAtom)

      installGracefulShutdownHandlers(
        renderer,
        async () => {
          stopDisplayViewController()
        },
        () => {
          const activeSessionId = getLastSessionId()
          if (!activeSessionId) {
            return
          }
          process.stdout.write(`\nResume this session with:\nmagnitude --resume ${activeSessionId}\n`)
        }
      )

      createRoot(renderer).render(
        <PlatformProvider platform={platform}>
          <RegistryProvider defaultIdleTTL={5000}>
            <AgentClientProvider tag={agentClientTag}>
              <DisplayViewControllerProvider>
                <CliApp
                  sessionStart={sessionStart}
                  initialPrompt={opts.prompt}
                  goal={opts.goal}
                  envAuth={resolveEnvAuth()}
                  forceLocalInferenceSetup={opts.setup ?? false}
                  sessionOptions={{
                    disableShellSafeguards: opts.disableShellSafeguards ?? false,
                    disableCwdSafeguards: opts.disableCwdSafeguards ?? false,
                    atifPath: opts.atif,
                    solo: opts.solo ?? false,
                    headless: false,
                    systemPromptOverride: opts.systemOverride,
                  }}
                />
              </DisplayViewControllerProvider>
            </AgentClientProvider>
          </RegistryProvider>
        </PlatformProvider>
      )
    })

  program.parse()
}

main()
