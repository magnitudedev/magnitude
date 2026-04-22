import { describe, it } from 'vitest'
import { shellValidator, multiToolValidator } from './helpers'

const YIELD = '<yield_user/>'
const CP = '</parameter>'
const CI = '</invoke>'
const CF = '</filter>'

// =============================================================================
// CATEGORY 1: POSITIVE — Must pass with both current and new grammar
// =============================================================================

describe('param-body deep confirm — basic positive', () => {
  it('normal close + sibling parameter', () => {
    multiToolValidator().passes(
      `<invoke tool="edit">\n<parameter name="path">src/foo.ts${CP}\n<parameter name="old">const x = 1${CP}\n${CI}\n${YIELD}`
    )
  })

  it('normal close + filter', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">ls${CP}\n<filter>$.stdout${CF}\n${CI}\n${YIELD}`
    )
  })

  it('normal close + invoke close', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">ls -la${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + < immediately + sibling parameter', () => {
    multiToolValidator().passes(
      `<invoke tool="edit">\n<parameter name="path">a${CP}<parameter name="old">b${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + < immediately + invoke close', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">ls${CP}${CI}\n${YIELD}`
    )
  })

  it('close + < immediately + filter', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">ls${CP}<filter>$.stdout${CF}\n${CI}\n${YIELD}`
    )
  })

  it('multiple parameters in sequence', () => {
    multiToolValidator().passes(
      `<invoke tool="edit">\n<parameter name="path">a.ts${CP}\n<parameter name="old">old${CP}\n<parameter name="new">new${CP}\n${CI}\n${YIELD}`
    )
  })

  it('empty parameter + valid continuation', () => {
    multiToolValidator().passes(
      `<invoke tool="edit">\n<parameter name="path">${CP}\n<parameter name="old">x${CP}\n${CI}\n${YIELD}`
    )
  })

  it('multi-line parameter content', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">line one\nline two\nline three${CP}\n${CI}\n${YIELD}`
    )
  })

  it('parameter content with HTML tags', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">echo "<div>hello</div>"${CP}\n${CI}\n${YIELD}`
    )
  })

  it('parameter content with < comparison', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">if [ $a < $b ]; then echo yes; fi${CP}\n${CI}\n${YIELD}`
    )
  })

  it('parameter content with </foo> (non-matching close)', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">echo </foo> done${CP}\n${CI}\n${YIELD}`
    )
  })

  it('parameter content with partial close </param>', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">see </param> here${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close with 1 space before newline', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">ls${CP} \n${CI}\n${YIELD}`
    )
  })

  it('close with 4 spaces before <', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">ls${CP}    ${CI}\n${YIELD}`
    )
  })

  it('parameter content with consecutive < chars', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">a << b <<< c${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + multiple newlines + invoke close', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">ls${CP}\n\n\n\n${CI}\n${YIELD}`
    )
  })

  it('close + tabs + newline + invoke close', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">ls${CP}\t\t\n${CI}\n${YIELD}`
    )
  })

})

// =============================================================================
// CATEGORY 2: NEW BEHAVIOR — Currently REJECTS, should ACCEPT after redesign
// These tests verify the new deep-confirmation behavior:
// close tag in content followed by non-structural text is treated as content.
// =============================================================================

describe('param-body deep confirm — new behavior (currently rejects)', () => {
  // Close tag in content followed by \n then non-structural text
  it('close tag in content + \\n + plain text, real close later', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">code${CP}\nhello world${CP}\n${CI}\n${YIELD}`
    )
  })

  // Close tag in content followed by <div> (non-structural)
  it('close tag in content + <div>, real close later', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">code${CP}<div>more${CP}\n${CI}\n${YIELD}`
    )
  })

  // Close tag followed by <python> (starts with p, not parameter)
  it('close tag in content + <python>, real close later', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">code${CP}\n<python>script${CP}\n${CI}\n${YIELD}`
    )
  })

  // Close tag followed by <parameterXYZ> (extra chars after 'parameter')
  it('close tag in content + <parameterXYZ>, real close later', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">code${CP}\n<parameterXYZ>stuff${CP}\n${CI}\n${YIELD}`
    )
  })

  // Close tag + \n + <div>
  it('close tag in content + \\n + <div>, real close later', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">code${CP}\n<div>html</div>${CP}\n${CI}\n${YIELD}`
    )
  })

  // Multiple false close tags, real one at end
  it('multiple false close tags, real one last', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">A${CP}\nB${CP}\nC${CP}\n${CI}\n${YIELD}`
    )
  })

  // Close tag followed by <foo> (unknown tag)
  it('close tag in content + <foo>, real close later', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">code${CP}\n<foo>bar${CP}\n${CI}\n${YIELD}`
    )
  })

  // Close tag followed by </div>
  it('close tag in content + </div>, real close later', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">code${CP}\n</div>more${CP}\n${CI}\n${YIELD}`
    )
  })

  // Close tag + 5 spaces (exceeds current MAX_TRAILING_WS=4)
  it('close tag + 5 spaces + valid continuation', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">ls${CP}     \n${CI}\n${YIELD}`
    )
  })

  // Close tag + /notinvoke
  it('close tag in content + </notinvoke>, real close later', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">code${CP}\n</notinvoke>more${CP}\n${CI}\n${YIELD}`
    )
  })

})

// =============================================================================
// CATEGORY 3: NEGATIVE — Should REJECT with both current and new grammar
// =============================================================================

describe('param-body deep confirm — always rejects', () => {
  // Under deep confirmation, yield isn't a valid continuation so the close
  // tag is rejected back to content. The grammar stays in freeform body mode —
  // every character is valid but the grammar never completes.
  // This is correct: no confirmation signal was received.
  it('parameter close + yield directly — no confirmation, stays in body mode', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">ls${CP}\n${YIELD}`
    )
  })

  it('invoke after yield', () => {
    shellValidator().rejects(
      `${YIELD}<invoke tool="shell">\n<parameter name="command">ls${CP}\n${CI}\n`
    )
  })

})

// =============================================================================
// CATEGORY 4: EDGE CASES
// =============================================================================

describe('param-body deep confirm — edge cases', () => {
  it('empty parameter value', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">${CP}\n${CI}\n${YIELD}`
    )
  })

  // KNOWN LIMITATION: content containing close tag + valid full continuation
  // prefix is falsely confirmed. Inherent to forward scanning.
  it('known limitation: close tag + valid continuation in content = false confirm', () => {
    // This is NOT a bug — it is the expected behavior of forward scanning.
    // The grammar sees valid structure and accepts it.
    multiToolValidator().passes(
      `<invoke tool="edit">\n<parameter name="path">a${CP}\n<parameter name="old">b${CP}\n${CI}\n${YIELD}`
    )
  })

})

// =============================================================================
// CATEGORY 5: FILTER BODY
// =============================================================================

describe('filter-body deep confirm', () => {
  it('normal filter close + invoke close', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">ls${CP}\n<filter>$.stdout${CF}\n${CI}\n${YIELD}`
    )
  })

  it('filter close + < immediately + invoke close', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<filter>$.stdout${CF}${CI}\n${YIELD}`
    )
  })

  it('filter close + sibling parameter', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<filter>$.stdout${CF}\n<parameter name="command">ls${CP}\n${CI}\n${YIELD}`
    )
  })

  // NEW BEHAVIOR for filter body
  it('filter content with false close, real close later', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<filter>$.stdout${CF}\nhello${CF}\n${CI}\n${YIELD}`
    )
  })

})
// =============================================================================
// CATEGORY 2b: LOOP-TRAP CASES — Realistic scenarios where model gets stuck
// =============================================================================

describe('param-body deep confirm — loop-trap cases (currently rejects)', () => {
  // The realistic trap: content mentions the close tag then continues with prose
  it('content has close tag + natural language paragraph', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">see ${CP}\nfor why this fails.\nEven more text here.${CP}\n${CI}\n${YIELD}`
    )
  })

  it('content has close tag followed by HTML comment', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">code${CP}\n<!-- a note -->${CP}\n${CI}\n${YIELD}`
    )
  })

  // Current grammar already handles this: space after > hits tw0, then 't' rejects back to s0
  it('content has close tag mid-sentence', () => {
    shellValidator().passes(
      `<invoke tool="shell">\n<parameter name="command">use ${CP} to end your parameter${CP}\n${CI}\n${YIELD}`
    )
  })
})

// =============================================================================
// CATEGORY 2c: PREFIX BOUNDARY PROBES — Pin down where confirmation DFA rejects
// =============================================================================

describe('param-body deep confirm — prefix boundary probes (currently rejects)', () => {
  // <p> — matches first char of 'parameter' but not second
  it('close + <p> in content', () => {
    shellValidator().passes(`<invoke tool="shell">\n<parameter name="command">x${CP}\n<p>hi</p>${CP}\n${CI}\n${YIELD}`)
  })

  // <par> — matches first 3 chars of 'parameter'
  it('close + <par> in content', () => {
    shellValidator().passes(`<invoke tool="shell">\n<parameter name="command">x${CP}\n<par>y${CP}\n${CI}\n${YIELD}`)
  })

  // <parameter> without name attr — full tag name but > instead of space
  it('close + <parameter> (no attr) in content', () => {
    shellValidator().passes(`<invoke tool="shell">\n<parameter name="command">x${CP}\n<parameter>hi${CP}\n${CI}\n${YIELD}`)
  })

  // <foo> — matches first char of 'filter' but not second
  it('close + <foo> in content (f prefix)', () => {
    shellValidator().passes(`<invoke tool="shell">\n<parameter name="command">x${CP}\n<foo>y${CP}\n${CI}\n${YIELD}`)
  })

  // </span> — starts with / but not /invoke
  it('close + </span> in content', () => {
    shellValidator().passes(`<invoke tool="shell">\n<parameter name="command">x${CP}\n</span>y${CP}\n${CI}\n${YIELD}`)
  })

  // </italic> — starts with /i but not /invoke
  it('close + </italic> in content', () => {
    shellValidator().passes(`<invoke tool="shell">\n<parameter name="command">x${CP}\n</italic>y${CP}\n${CI}\n${YIELD}`)
  })
})

// =============================================================================
// CATEGORY 4b: KNOWN LIMITATION — explicit false-confirm documentation
// =============================================================================

describe('param-body deep confirm — known limitation (false confirm)', () => {
  it('content with full valid continuation prefix gets false-confirmed', () => {
    // Model wants to write docs containing the close tag + valid continuation
    // as literal content. The grammar accepts it as valid STRUCTURE — the path
    // parameter only gets 'see ' and a second parameter opens unexpectedly.
    // This is the documented limitation of forward scanning.
    multiToolValidator().passes(
      `<invoke tool="edit">\n<parameter name="path">see ${CP}\n<parameter name="old">...${CP}\n${CI}\n${YIELD}`
    )
  })
})

// =============================================================================
// CATEGORY 5b: FILTER BODY PREFIX BOUNDARIES
// =============================================================================

describe('filter-body deep confirm — prefix boundaries (currently rejects)', () => {
  it('filter close + <p> (not parameter) in content', () => {
    shellValidator().passes(`<invoke tool="shell">\n<filter>$.a${CF}\n<p>hi${CF}\n${CI}\n${YIELD}`)
  })

  it('filter close + </notinvoke> in content', () => {
    shellValidator().passes(`<invoke tool="shell">\n<filter>$.a${CF}\n</notinvoke>more${CF}\n${CI}\n${YIELD}`)
  })
})

// =============================================================================
// CATEGORY 2d: HIGHER WHITESPACE PROBES
// =============================================================================

describe('param-body deep confirm — extended whitespace (currently rejects)', () => {
  it('close + 10 spaces + valid continuation', () => {
    shellValidator().passes(`<invoke tool="shell">\n<parameter name="command">ls${CP}          \n${CI}\n${YIELD}`)
  })

  // 4 chars of whitespace fits within current MAX_TRAILING_WS=4
  it('close + tab-space-tab-space + valid continuation', () => {
    shellValidator().passes(`<invoke tool="shell">\n<parameter name="command">ls${CP}\t \t \n${CI}\n${YIELD}`)
  })
})
