import { createCliRenderer } from '@opentui/core'
import { createRoot } from '@opentui/react'
import { appendFileSync, writeFileSync } from 'fs'
import { useRef, useEffect } from 'react'

writeFileSync('/tmp/paste-test.log', '')
const log = (msg: string) => appendFileSync('/tmp/paste-test.log', msg + '\n')

log('SCRIPT STARTED')

function TestApp() {
  const scrollBoxRef = useRef<any>(null)

  useEffect(() => {
    log('useEffect - focusing scrollbox')
    log('scrollBoxRef.current: ' + !!scrollBoxRef.current)
    log('has focus: ' + typeof scrollBoxRef.current?.focus)
    log('_pasteListener BEFORE focus: ' + !!scrollBoxRef.current?._pasteListener)
    scrollBoxRef.current?.focus?.()
    log('_focused after: ' + scrollBoxRef.current?._focused)
    log('_pasteListener AFTER focus: ' + !!scrollBoxRef.current?._pasteListener)
    log('pasteHandler: ' + !!scrollBoxRef.current?.pasteHandler)

    // Check the keyHandler's renderableHandlers
    const ctx = scrollBoxRef.current?.ctx
    const keyHandler = ctx?._internalKeyInput || ctx?._keyHandler
    log('keyHandler: ' + !!keyHandler)
    log('renderableHandlers: ' + !!keyHandler?.renderableHandlers)
    const pasteHandlers = keyHandler?.renderableHandlers?.get('paste')
    log('paste handlers set: ' + !!pasteHandlers)
    log('paste handlers size: ' + (pasteHandlers?.size ?? 0))
  }, [])

  return (
    <scrollbox
      ref={scrollBoxRef}
      onPaste={(event: any) => {
        log('SCROLLBOX onPaste: ' + event.text)
      }}
      style={{ flexGrow: 1 }}
    >
      <text>CMD+V to test (no manual bracketed paste enable)</text>
    </scrollbox>
  )
}

async function main() {
  log('MAIN STARTED')
  const renderer = await createCliRenderer({
    backgroundColor: 'transparent',
  }) as any

  log('renderer created')

  // Listen on _stdinBuffer for comparison
  renderer._stdinBuffer?.on('paste', (text: string) => {
    log('_stdinBuffer paste: ' + text.substring(0, 50))
  })

  // Intercept the keyHandler's emit FIRST (before patching processPaste)
  const keyHandler = renderer._keyHandler
  const originalEmit = keyHandler.emit.bind(keyHandler)
  keyHandler.emit = (event: string, ...args: any[]) => {
    if (event === 'paste') {
      log('keyHandler.emit paste called')
      const pasteEvent = args[0]
      log('  defaultPrevented: ' + pasteEvent?.defaultPrevented)
      log('  propagationStopped: ' + pasteEvent?.propagationStopped)
      log('  text: ' + pasteEvent?.text?.substring(0, 30))
      const result = originalEmit(event, ...args)
      log('  emit returned: ' + result)
      return result
    }
    return originalEmit(event, ...args)
  }

  // Now replace processPaste to call our patched emit
  keyHandler.processPaste = (data: string) => {
    log('processPaste called with: ' + data.substring(0, 30))
    try {
      // Inline the processPaste logic so it uses our patched emit
      const cleanedData = (Bun as any).stripANSI(data)
      log('cleanedData: ' + cleanedData.substring(0, 30))
      // Get PasteEvent class - need to import or access it
      const PasteEvent = (keyHandler as any).constructor.prototype.constructor.PasteEvent ||
                         Object.getPrototypeOf(keyHandler).constructor.PasteEvent
      log('PasteEvent class: ' + !!PasteEvent)
      // Just call emit directly with a simple object for now
      keyHandler.emit('paste', { text: cleanedData, defaultPrevented: false, propagationStopped: false })
      log('processPaste completed')
    } catch (e: any) {
      log('processPaste threw: ' + e.message + ' ' + e.stack)
    }
  }

  createRoot(renderer).render(<TestApp />)
}

main()
