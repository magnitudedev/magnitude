process.env.BAML_LOG = 'off';

import fs from 'fs'
import { createCliRenderer } from '@opentui/core'
import { createRoot } from '@opentui/react'
import { Command } from '@commander-js/extra-typings'
import { createProviderClient } from '@magnitudedev/providers'
import { createStorageClient } from '@magnitudedev/storage'
import { MAGNITUDE_SLOTS, type MagnitudeSlot } from '@magnitudedev/agent'
import { App, type SessionStart } from './app'
import { initThemeStore, useThemeStateStore } from './hooks/use-theme'
import { CLI_VERSION } from './version'
import { ProviderRuntimeProvider } from './providers/provider-runtime'
import { StorageProvider } from './providers/storage-provider'
import { isLightBackground } from './utils/theme'
import { installGracefulShutdownHandlers } from './utils/graceful-shutdown'

import { runOneshot } from './oneshot'

async function main() {
  // Initialize theme store before rendering (defaults to dark)
  initThemeStore()

  const program = new Command()
    .name('magnitude')
    .version(CLI_VERSION)
    .option('--resume [id]', 'Resume the most recent chat session or a specific session by ID')
    .option('--debug', 'Enable debug mode with debug panel')

    .option('--oneshot [prompt]', 'Run autonomous oneshot task and exit on completion')
    .option('--provider <id>', 'Provider ID for oneshot mode (e.g. anthropic, openai)')
    .option('--model <id>', 'Model ID for oneshot mode')
    .option('--disable-shell-safeguards', 'Disable shell command classification safeguards for this oneshot run')
    .option('--disable-cwd-safeguards', 'Disable working-directory boundary safeguards for this oneshot run')
    .argument('[prompt]')
    .action(async (promptArg, opts) => {
      if (opts.oneshot !== undefined) {
        if (opts.resume !== undefined) {
          console.error('--resume and --oneshot cannot be used together')
          process.exit(1)
        }
        const prompt = typeof opts.oneshot === 'string' ? opts.oneshot : promptArg
        await runOneshot({
          prompt,
          providerId: opts.provider,
          modelId: opts.model,
          debug: opts.debug ?? false,
          disableShellSafeguards: opts.disableShellSafeguards ?? false,
          disableCwdSafeguards: opts.disableCwdSafeguards ?? false,
        })
        return
      }

      const renderer = await createCliRenderer({
        exitOnCtrlC: false, // We handle Ctrl+C manually for two-tap exit
      })

      // Non-blocking: detect terminal background, switch to light theme if needed
      renderer.getPalette({ timeout: 1000 }).then((colors) => {
        if (colors?.defaultBackground) {
          useThemeStateStore.getState().setTerminalDetectedBg(colors.defaultBackground)
          if (isLightBackground(colors.defaultBackground)) {
            useThemeStateStore.getState().setThemeName('light')
          }
        }
      }).catch(() => {})

      let clientRef: { dispose: () => Promise<void> } | null = null
      let activeSessionId: string | null = null

      installGracefulShutdownHandlers(
        renderer,
        async () => {
          await clientRef?.dispose()
        },
        () => {
          if (!activeSessionId) {
            return
          }
          fs.writeSync(1, `\nResume this session with:\nmagnitude --resume ${activeSessionId}\n`)
        }
      )

      const storage = await createStorageClient({ cwd: process.cwd(), currentVersion: CLI_VERSION })
      const providerRuntime = await createProviderClient<MagnitudeSlot>({ slots: MAGNITUDE_SLOTS })
      const sessionStart: SessionStart = opts.resume === undefined
        ? { _tag: 'new' }
        : opts.resume === true
          ? { _tag: 'latest' }
          : { _tag: 'resume', sessionId: opts.resume }

      createRoot(renderer).render(
        <StorageProvider client={storage}>
          <ProviderRuntimeProvider runtime={providerRuntime}>
            <App
              sessionStart={sessionStart}
              debug={opts.debug ?? false}
              onClientReady={(client) => {
                clientRef = client
              }}
              onSessionId={(id) => {
                activeSessionId = id
              }}
            />
          </ProviderRuntimeProvider>
        </StorageProvider>
      )
    })

  program
    .command('serve')
    .description('Start the magnitude API server')
    .option('-p, --port <port>', 'Port to listen on', '8080')
    .option('--host <host>', 'Host to bind to', '127.0.0.1')
    .option('--token <token>', 'Bearer token for authentication')
    .option('--debug', 'Enable debug mode')
    .action(async (options) => {
      const { startServer } = await import('./serve')
      await startServer({
        port: parseInt(options.port),
        host: options.host,
        token: options.token ?? process.env.MAGNITUDE_SERVE_TOKEN,
        debug: options.debug ?? false
      })
    })

  program
    .command('skills')
    .description('Manage skills')
    .addCommand(
      new Command('reset')
        .description('Reset global skills to built-in defaults')
        .action(async () => {
          const fs = await import('fs')
          const path = await import('path')
          const os = await import('os')
          const readline = await import('readline')

          const globalSkillsDir = path.join(os.homedir(), '.magnitude', 'skills')

          console.log('WARNING: This will overwrite ALL skills in ~/.magnitude/skills/ with the built-in defaults.')
          console.log('Any custom global skills you have created will be lost.')
          console.log()

          const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
          })

          const answer = await new Promise<string>((resolve) => {
            rl.question('Type "yes" to proceed or "no" to cancel: ', (ans) => {
              rl.close()
              resolve(ans.trim())
            })
          })

          if (answer === 'yes') {
            fs.rmSync(globalSkillsDir, { recursive: true, force: true })
            const { initGlobalSkills } = await import('@magnitudedev/skills')
            await initGlobalSkills()
            console.log('Global skills reset to built-in defaults.')
          } else {
            console.log('Cancelled. No changes were made.')
            process.exit(0)
          }
        })
    )

  program.parse()
}

main()

