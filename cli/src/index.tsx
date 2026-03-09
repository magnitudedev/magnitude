import { createCliRenderer } from '@opentui/core'
import { createRoot } from '@opentui/react'
import { Command } from '@commander-js/extra-typings'
import { App } from './app'
import { initThemeStore, useThemeStateStore } from './hooks/use-theme'
import { isLightBackground } from './utils/theme'
import { installGracefulShutdownHandlers } from './utils/graceful-shutdown'
import { useAltKeywords } from '@magnitudedev/xml-act'

async function main() {
  // Initialize theme store before rendering (defaults to dark)
  initThemeStore()

  const program = new Command()
    .name('magnitude')
    .version('0.0.1')
    .option('--resume', 'Resume the most recent chat session')
    .option('--debug', 'Enable debug mode with debug panel')
    .option('--alt-keywords', 'Use alternate XML keywords (magniactions/magnithink) for self-development')
    .action(async (opts) => {
      if (opts.altKeywords) {
        useAltKeywords()
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

      installGracefulShutdownHandlers(renderer)
      createRoot(renderer).render(<App resume={opts.resume ?? false} debug={opts.debug ?? false} />)
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

  program.parse()
}

main()

