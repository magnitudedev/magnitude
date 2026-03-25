export type ScannerSignal =
  | {
      readonly type: 'open'
      readonly tagName: string
      readonly attrs: ReadonlyMap<string, string>
      readonly afterNewline: boolean
      readonly raw?: string
    }
  | {
      readonly type: 'close'
      readonly tagName: string
      readonly afterNewline: boolean
      readonly raw?: string
    }
  | {
      readonly type: 'selfClose'
      readonly tagName: string
      readonly attrs: ReadonlyMap<string, string>
      readonly afterNewline: boolean
      readonly raw?: string
    }
  | { readonly type: 'content'; readonly text: string }

export interface Scanner {
  push(chunk: string): void
  end(): void
}

type TagPhase =
  | 'name'
  | 'attrs'
  | 'attrKey'
  | 'attrAfterKey'
  | 'attrBeforeValue'
  | 'attrValueQuoted'
  | 'attrValueUnquoted'

type ActiveTag = {
  raw: string
  savedAfterNewline: boolean
  isClose: boolean
  name: string
  attrs: Map<string, string>
  phase: TagPhase
  pendingSelfClose: boolean
  attrKey: string
  attrValue: string
  attrQuote: '"' | "'" | null
}

type FenceState = {
  inFence: boolean
  fenceChar: '`' | '~' | null
  fenceLength: number
  lineStart: boolean
  strip: boolean
  partialPrefix: string
}

const CDATA_OPEN = '<' + '!' + '[CDATA['
const CDATA_CLOSE = ']' + ']' + '>'

function isWhitespace(ch: string): boolean {
  return ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r'
}

function isNameStart(ch: string): boolean {
  return /[a-zA-Z_]/.test(ch)
}

function isNameContinue(ch: string): boolean {
  return /[a-zA-Z0-9_.-]/.test(ch)
}

function withAfterNewline(signal: ScannerSignal): ScannerSignal {
  switch (signal.type) {
    case 'open': return { ...signal, afterNewline: true }
    case 'close': return { ...signal, afterNewline: true }
    case 'selfClose': return { ...signal, afterNewline: true }
    case 'content': return signal
  }
}

export function createScanner(onSignal: (signal: ScannerSignal) => void): Scanner {
  let contentBuffer = ''
  let afterNewline = true

  const fence: FenceState = {
    inFence: false,
    fenceChar: null,
    fenceLength: 0,
    lineStart: true,
    strip: false,
    partialPrefix: '',
  }

  let activeTag: ActiveTag | null = null
  let pendingTag: { signal: ScannerSignal; allowEofAsNewline: boolean } | null = null

  let cdataBuffer: string | null = null

  function updateContentFlags(text: string): void {
    for (const ch of text) {
      if (ch === '\n') {
        afterNewline = true
        fence.lineStart = true
      } else {
        afterNewline = false
        fence.lineStart = false
      }
    }
  }

  function flushContent(): void {
    if (contentBuffer.length === 0) return
    const text = contentBuffer
    contentBuffer = ''
    onSignal({ type: 'content', text })
    updateContentFlags(text)
  }

  function failTagAsContent(): void {
    if (!activeTag) return
    contentBuffer += activeTag.raw
    activeTag = null
  }

  function emitTag(tag: ActiveTag): void {
    flushContent()
    const base = tag.savedAfterNewline
    const signal: ScannerSignal = tag.isClose
      ? { type: 'close', tagName: tag.name, afterNewline: base, raw: tag.raw }
      : tag.pendingSelfClose
        ? {
            type: 'selfClose',
            tagName: tag.name,
            attrs: new Map(tag.attrs),
            afterNewline: base,
            raw: tag.raw,
          }
        : {
            type: 'open',
            tagName: tag.name,
            attrs: new Map(tag.attrs),
            afterNewline: base,
            raw: tag.raw,
          }

    const shouldDefer =
      !signal.afterNewline && (
        (signal.type === 'close' && signal.tagName === 'think') ||
        (signal.type === 'open' && signal.tagName === 'think') ||
        (signal.type === 'close' && signal.tagName === 'actions')
      )

    if (shouldDefer) {
      pendingTag = {
        signal,
        allowEofAsNewline: signal.type === 'close' && signal.tagName === 'think',
      }
    } else {
      onSignal(signal)
    }

    afterNewline = false
  }

  function startTag(): void {
    activeTag = {
      raw: '<',
      savedAfterNewline: afterNewline,
      isClose: false,
      name: '',
      attrs: new Map(),
      phase: 'name',
      pendingSelfClose: false,
      attrKey: '',
      attrValue: '',
      attrQuote: null,
    }
  }

  function maybeFenceRun(chunk: string, index: number): number {
    if (!fence.lineStart) return index
    const ch = chunk[index]
    if (ch !== '`' && ch !== '~') return index

    let j = index
    while (j < chunk.length && chunk[j] === ch) j++
    const runLen = j - index
    if (runLen < 3) {
      if (j === chunk.length) {
        fence.partialPrefix = chunk.slice(index, j)
        return j
      }
      return index
    }

    let k = j
    while (k < chunk.length && chunk[k] !== '\n' && chunk[k] !== '\r') k++

    if (!fence.inFence && k === chunk.length) {
      fence.partialPrefix = chunk.slice(index, k)
      return k
    }

    const info = chunk.slice(j, k).trim().toLowerCase()

    if (!fence.inFence) {
      fence.inFence = true
      fence.fenceChar = ch
      fence.fenceLength = runLen
      fence.strip = info.startsWith('xml')
      if (fence.strip) {
        if (k < chunk.length && chunk[k] === '\r') k++
        if (k < chunk.length && chunk[k] === '\n') k++
        afterNewline = true
        fence.lineStart = true
        return k
      }
    } else if (fence.fenceChar === ch && runLen >= fence.fenceLength) {
      const strip = fence.strip
      fence.inFence = false
      fence.fenceChar = null
      fence.fenceLength = 0
      fence.strip = false
      if (strip) {
        if (k < chunk.length && chunk[k] === '\r') k++
        if (k < chunk.length && chunk[k] === '\n') k++
        afterNewline = true
        fence.lineStart = true
        return k
      }
    }

    contentBuffer += chunk.slice(index, j)
    afterNewline = false
    fence.lineStart = false
    return j
  }

  function finalizeBooleanAttr(tag: ActiveTag): void {
    if (tag.attrKey.length > 0) {
      tag.attrs.set(tag.attrKey, '')
      tag.attrKey = ''
    }
  }

  function finalizeAttrValue(tag: ActiveTag): void {
    tag.attrs.set(tag.attrKey, tag.attrValue)
    tag.attrKey = ''
    tag.attrValue = ''
    tag.attrQuote = null
  }

  function processTagChar(ch: string): void {
    const tag = activeTag!

    if (ch === '\n' || ch === '\r') {
      failTagAsContent()
      contentBuffer += ch
      afterNewline = true
      fence.lineStart = true
      return
    }

    if (ch === '<' && tag.phase !== 'attrValueQuoted') {
      failTagAsContent()
      startTag()
      return
    }

    tag.raw += ch

    if (tag.phase === 'name') {
      if (tag.raw.startsWith('<!')) {
        if (CDATA_OPEN.startsWith(tag.raw)) {
          if (tag.raw === CDATA_OPEN) {
            activeTag = null
            cdataBuffer = ''
          }
          return
        }
        failTagAsContent()
        return
      }

      if (tag.raw.length === 2 && ch === '/') {
        tag.isClose = true
        return
      }

      if (tag.raw.length === 2 && ch === '!') {
        return
      }

      const firstNamePos = tag.isClose ? 2 : 1
      const namePos = tag.raw.length - 1 - firstNamePos
      if (namePos < 0) return

      if (namePos === 0) {
        if (!isNameStart(ch)) failTagAsContent()
        else tag.name += ch
        return
      }

      if (isNameContinue(ch)) {
        tag.name += ch
        return
      }

      if (isWhitespace(ch)) {
        tag.phase = 'attrs'
        return
      }

      if (ch === '>') {
        emitTag(tag)
        activeTag = null
        return
      }

      if (ch === '/' && !tag.isClose) {
        tag.pendingSelfClose = true
        tag.phase = 'attrs'
        return
      }

      failTagAsContent()
      return
    }

    if (tag.phase === 'attrs') {
      if (isWhitespace(ch)) return

      if (ch === '>') {
        emitTag(tag)
        activeTag = null
        return
      }

      if (ch === '/' && !tag.isClose) {
        tag.pendingSelfClose = true
        return
      }

      if (tag.pendingSelfClose) {
        failTagAsContent()
        return
      }

      if (!isNameStart(ch)) {
        failTagAsContent()
        return
      }

      tag.attrKey = ch
      tag.phase = 'attrKey'
      return
    }

    if (tag.phase === 'attrKey') {
      if (isNameContinue(ch)) {
        tag.attrKey += ch
        return
      }

      if (isWhitespace(ch)) {
        tag.phase = 'attrAfterKey'
        return
      }

      if (ch === '=') {
        tag.phase = 'attrBeforeValue'
        return
      }

      if (ch === '>') {
        finalizeBooleanAttr(tag)
        emitTag(tag)
        activeTag = null
        return
      }

      if (ch === '/' && !tag.isClose) {
        finalizeBooleanAttr(tag)
        tag.pendingSelfClose = true
        tag.phase = 'attrs'
        return
      }

      failTagAsContent()
      return
    }

    if (tag.phase === 'attrAfterKey') {
      if (isWhitespace(ch)) return

      if (ch === '=') {
        tag.phase = 'attrBeforeValue'
        return
      }

      if (ch === '>') {
        finalizeBooleanAttr(tag)
        emitTag(tag)
        activeTag = null
        return
      }

      if (ch === '/' && !tag.isClose) {
        finalizeBooleanAttr(tag)
        tag.pendingSelfClose = true
        tag.phase = 'attrs'
        return
      }

      // key with no value, then start next key
      finalizeBooleanAttr(tag)
      if (!isNameStart(ch)) {
        failTagAsContent()
        return
      }
      tag.attrKey = ch
      tag.phase = 'attrKey'
      return
    }

    if (tag.phase === 'attrBeforeValue') {
      if (isWhitespace(ch)) return

      if (ch === '"' || ch === "'") {
        tag.attrQuote = ch
        tag.attrValue = ''
        tag.phase = 'attrValueQuoted'
        return
      }

      if (ch === '>') {
        tag.attrValue = ''
        finalizeAttrValue(tag)
        emitTag(tag)
        activeTag = null
        return
      }

      if (ch === '/' && !tag.isClose) {
        tag.attrValue = ''
        finalizeAttrValue(tag)
        tag.pendingSelfClose = true
        tag.phase = 'attrs'
        return
      }

      tag.attrValue = ch
      tag.phase = 'attrValueUnquoted'
      return
    }

    if (tag.phase === 'attrValueQuoted') {
      if (ch === tag.attrQuote) {
        finalizeAttrValue(tag)
        tag.phase = 'attrs'
        return
      }
      tag.attrValue += ch
      return
    }

    if (tag.phase === 'attrValueUnquoted') {
      if (isWhitespace(ch)) {
        finalizeAttrValue(tag)
        tag.phase = 'attrs'
        return
      }

      if (ch === '>') {
        finalizeAttrValue(tag)
        emitTag(tag)
        activeTag = null
        return
      }

      if (ch === '/' && !tag.isClose) {
        finalizeAttrValue(tag)
        tag.pendingSelfClose = true
        tag.phase = 'attrs'
        return
      }

      tag.attrValue += ch
    }
  }

  return {
    push(chunk: string): void {
      let input = chunk
      if (fence.partialPrefix.length > 0) {
        input = fence.partialPrefix + input
        fence.partialPrefix = ''
      }

      let i = 0
      while (i < input.length) {
        if (cdataBuffer !== null) {
          const closeIdx = input.indexOf(CDATA_CLOSE, i)
          if (closeIdx === -1) {
            cdataBuffer += input.slice(i)
            i = input.length
            continue
          }
          cdataBuffer += input.slice(i, closeIdx)
          contentBuffer += cdataBuffer
          cdataBuffer = null
          i = closeIdx + CDATA_CLOSE.length
          continue
        }

        if (activeTag) {
          processTagChar(input[i])
          i++
          continue
        }

        const jump = maybeFenceRun(input, i)
        if (jump !== i) {
          i = jump
          continue
        }

        const ch = input[i]

        if (pendingTag) {
          const useNewline = ch === '\n' || ch === '\r'
          onSignal(useNewline ? withAfterNewline(pendingTag.signal) : pendingTag.signal)
          pendingTag = null
        }

        if (ch === '<' && (!fence.inFence || fence.strip)) {
          const next = i + 1 < input.length ? input[i + 1] : ''
          if (next === '' || next === '/' || next === '!' || isNameStart(next)) {
            flushContent()
            startTag()
            i++
            continue
          }
        }

        contentBuffer += ch
        if (ch === '\n') {
          afterNewline = true
          fence.lineStart = true
        } else {
          afterNewline = false
          fence.lineStart = false
        }
        i++
      }
      if (!activeTag && cdataBuffer === null && fence.partialPrefix.length === 0) {
        flushContent()
      }

    },

    end(): void {
      if (activeTag) {
        if (!activeTag.isClose && activeTag.name.length > 0 && activeTag.phase !== 'attrValueQuoted') {
          emitTag(activeTag)
          activeTag = null
        } else {
          failTagAsContent()
        }
      }
      if (pendingTag) {
        const sig = pendingTag.allowEofAsNewline
          ? withAfterNewline(pendingTag.signal)
          : pendingTag.signal
        onSignal(sig)
        pendingTag = null
      }
      if (cdataBuffer !== null) {
        contentBuffer += CDATA_OPEN + cdataBuffer
        cdataBuffer = null
      }
      flushContent()
    },
  }
}
