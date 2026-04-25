import { describe, it, expect } from 'vitest'
import { shellValidator, multiToolValidator, buildValidator, SHELL_TOOL, MULTI_PARAM_TOOL } from './helpers'

const YIELD = '<magnitude:yield_user/>'
const CP = '</magnitude:parameter>'
const CI = '</magnitude:invoke>'
const CF = '</magnitude:filter>'
const OP = '<magnitude:parameter'
const OF = '<magnitude:filter>'
const OI = '<magnitude:invoke>'

// =============================================================================
// CATEGORY 1: BASIC — Must pass with any correct grammar
// =============================================================================

describe('greedy param-body — basic', () => {
  it('simple param', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls -la${CP}\n${CI}\n${YIELD}`
    )
  })

  it('empty param body', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">${CP}\n${CI}\n${YIELD}`
    )
  })

  it('multi-line content', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">line one\nline two\nline three${CP}\n${CI}\n${YIELD}`
    )
  })

  it('multiple params', () => {
    multiToolValidator().passes(
      `<magnitude:invoke tool="edit">\n${OP} name="path">a.ts${CP}\n${OP} name="old">x${CP}\n${OP} name="new">y${CP}\n${CI}\n${YIELD}`
    )
  })

  it('param + filter', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls${CP}\n${OF}$.stdout${CF}\n${CI}\n${YIELD}`
    )
  })

  it('close immediately followed by next param', () => {
    multiToolValidator().passes(
      `<magnitude:invoke tool="edit">\n${OP} name="path">a${CP}${OP} name="old">b${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close immediately followed by invoke close', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls${CP}${CI}\n${YIELD}`
    )
  })
})

// =============================================================================
// CATEGORY 2: SIMPLE EMBEDDED CLOSE TAGS
// Close tag in content followed by non-structural text.
// These pass with deep-confirmation too (non-structural continuation = reject to content).
// =============================================================================

describe('greedy param-body — simple embedded close tags', () => {
  it('close tag + space + word', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo ${CP} done${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close tag + newline + plain text', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">code${CP}\nhello world${CP}\n${CI}\n${YIELD}`
    )
  })

  it('two embedded close tags', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">A${CP}B${CP}C${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close tag followed by HTML', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">code${CP}<div>more</div>${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close tag mid-sentence', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">use ${CP} to end${CP}\n${CI}\n${YIELD}`
    )
  })
})

// =============================================================================
// CATEGORY 3: GREEDY — EMBEDDED CLOSE + VALID CONTINUATION PREFIX
// These are the critical tests. The content contains a close tag followed by
// something that LOOKS like valid structural continuation. Deep-confirmation
// commits here (false positive). Greedy last-match treats it as content.
// =============================================================================

describe('greedy param-body — false commit scenarios (MUST FAIL with eager/deep, PASS with greedy)', () => {

  // --- Close tag + valid parameter open ---

  it('close + param open with name attr + more content + real close', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">START${CP}\n${OP} name="x">MIDDLE${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + param open (no whitespace) + content + real close', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">A${CP}${OP} name="y">B${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + whitespace + param open + content + real close', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">X${CP}  \n${OP} name="z">Y${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + param open + close + param open + real close (double fake)', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">a${CP}\n${OP} name="b">c${CP}\n${OP} name="d">e${CP}\n${CI}\n${YIELD}`
    )
  })

  // --- Close tag + valid invoke close ---

  it('close + invoke close + more content + real close', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">fake${CP}${CI}real${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + newline + invoke close + more content + real close', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">fake${CP}\n${CI}\nmore${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + invoke close + newline + text + real close', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">X${CP}${CI}\nY${CP}\n${CI}\n${YIELD}`
    )
  })

  // --- Close tag + valid filter open ---

  it('close + filter open + content + real close', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">fake${CP}${OF}notreal${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + newline + filter open + content + real close', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">fake${CP}\n${OF}notreal${CP}\n${CI}\n${YIELD}`
    )
  })

  // --- Full fake structural sequences ---

  it('close + invoke close + garbage + real close + real invoke close', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">prefix ${CP}\n${CI}\ngarbage suffix${CP}\n${CI}\n${YIELD}`
    )
  })

  it('entire fake invoke block in content', () => {
    // Content contains what looks like a complete param close + invoke close + new invoke
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo ${CP}\n${CI}\n${OI} tool="fake">\n${OP} name="x">y${CP}\n${CI}\nstill content${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + param open + close + invoke close + text + real close', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">A${CP}\n${OP} name="b">C${CP}\n${CI}\nD${CP}\n${CI}\n${YIELD}`
    )
  })

  // --- Close + invoke close + top-level structural elements ---

  it('close + invoke close + yield (exits everything)', () => {
    // Content: "fake</param>\n</magnitude:invoke>\n<magnitude:yield_user/>"
    // Deep-confirm commits, exits invoke, yield ends the turn — orphans the real content
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">fake${CP}\n${CI}\n${YIELD}still here${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + invoke close + reason block + real close', () => {
    // Content contains what looks like: close param, close invoke, start reason
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">fake${CP}\n${CI}\n<magnitude:reason about="x">thought</magnitude:reason>\nreal${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + invoke close + message block + real close', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">fake${CP}\n${CI}\n<magnitude:message to="user">hi</magnitude:message>\nreal${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + invoke close + new invoke + real close', () => {
    // Content contains an entire fake invoke block
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">fake${CP}\n${CI}\n<magnitude:invoke tool="other">\n${OP} name="x">y${CP}\n${CI}\nreal${CP}\n${CI}\n${YIELD}`
    )
  })

  // --- Variations of invoke-close positioning ---

  it('close + space + invoke close + content', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">X${CP} ${CI}Y${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + tab + invoke close + content', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">X${CP}\t${CI}Y${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + multiple newlines + invoke close + content', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">X${CP}\n\n\n${CI}\nY${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + invoke close repeated twice in content', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">A${CP}${CI}B${CP}${CI}C${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + invoke close + spaces + text + real close', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">X${CP}${CI}   still content${CP}\n${CI}\n${YIELD}`
    )
  })

  // --- Filter body with invoke close ---

  it('filter close + invoke close + text + real filter close', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n${OF}data${CF}\n${CI}\nmore${CF}\n${CI}\n${YIELD}`
    )
  })

  it('filter close + invoke close + param open + real filter close', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n${OF}data${CF}${CI}${OP} name="x">fake${CF}\n${CI}\n${YIELD}`
    )
  })

  // --- Realistic code that triggers false commit ---

  it('writing XML documentation about invoke close syntax', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo "close with ${CP} then ${CI} to finish"${CP}\n${CI}\n${YIELD}`
    )
  })

  it('sed replacing invoke close tags', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">sed 's/${CP}\\n${CI}/replaced/g' file.xml${CP}\n${CI}\n${YIELD}`
    )
  })

  it('grep searching for close + invoke pattern', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">grep -P '${CP}\\s*${CI}' src/*.ts${CP}\n${CI}\n${YIELD}`
    )
  })

  it('python code building XML with close tags', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">python3 -c "xml = '${CP}\\n${CI}'; print(xml)"${CP}\n${CI}\n${YIELD}`
    )
  })

  it('heredoc containing close sequence', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">cat << 'EOF'\n${CP}\n${CI}\nEOF${CP}\n${CI}\n${YIELD}`
    )
  })
})

// =============================================================================
// CATEGORY 4: REALISTIC CONTENT — Code that naturally contains close-tag-like text
// =============================================================================

describe('greedy param-body — realistic content with close-tag-like text', () => {

  it('sed command replacing XML tags', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">sed 's/${CP}/${CF}/g' file.xml${CP}\n${CI}\n${YIELD}`
    )
  })

  it('echo command outputting XML — rejected (magnitude open in body must be escaped)', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo "${OP} name=\\"x\\">val${CP}"${CP}\n${CI}\n${YIELD}`
    )
  })

  it('grep for close tag pattern', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">grep '${CP}' src/*.ts${CP}\n${CI}\n${YIELD}`
    )
  })

  it('documentation about the XML format itself', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo "Use ${CP} to close a param and ${CI} to close an invoke"${CP}\n${CI}\n${YIELD}`
    )
  })

  it('python code with XML string — rejected (magnitude open in body must be escaped)', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">python3 -c "print('${OP} name=x>val${CP}')"${CP}\n${CI}\n${YIELD}`
    )
  })

  it('code comment mentioning close tag', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">cat << EOF\n# close with ${CP}\n# then ${CI}\nEOF${CP}\n${CI}\n${YIELD}`
    )
  })
})

// =============================================================================
// CATEGORY 5: FILTER BODY GREEDY
// =============================================================================

describe('greedy filter-body — embedded close tags', () => {
  it('basic filter', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls${CP}\n${OF}$.stdout${CF}\n${CI}\n${YIELD}`
    )
  })

  it('filter with embedded close', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n${OF}$.stdout${CF}\nhello${CF}\n${CI}\n${YIELD}`
    )
  })

  it('filter close + param open (valid continuation) + content + real close', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n${OF}data${CF}\n${OP} name="x">fake${CF}\n${CI}\n${YIELD}`
    )
  })

  it('filter close + invoke close + content + real close', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n${OF}data${CF}${CI}more${CF}\n${CI}\n${YIELD}`
    )
  })
})

// =============================================================================
// CATEGORY 6: WHITESPACE VARIATIONS
// =============================================================================

describe('greedy param-body — whitespace edge cases', () => {
  it('close + 5 spaces + continuation', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls${CP}     \n${CI}\n${YIELD}`
    )
  })

  it('close + 10 spaces + continuation', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls${CP}          \n${CI}\n${YIELD}`
    )
  })

  it('close + mixed whitespace + continuation', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls${CP}\t \t \n${CI}\n${YIELD}`
    )
  })

  it('close + multiple newlines + continuation', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls${CP}\n\n\n\n${CI}\n${YIELD}`
    )
  })

  it('close + newline + spaces + continuation', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls${CP}\n   ${CI}\n${YIELD}`
    )
  })
})

// =============================================================================
// CATEGORY 7: FULL TURN INTEGRATION
// =============================================================================

describe('greedy param-body — full turn', () => {
  it('reason + message + invoke with embedded close', () => {
    shellValidator().passes(
      `<magnitude:reason about="test">thinking</magnitude:reason>\n` +
      `<magnitude:message to="user">hello</magnitude:message>\n` +
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo ${CP} works${CP}\n${CI}\n${YIELD}`
    )
  })

  it('multiple invokes, second has embedded close', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls${CP}\n${CI}\n` +
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo ${CP} test${CP}\n${CI}\n${YIELD}`
    )
  })

  it('invoke with greedy content + second invoke', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">fake${CP}${CI}real${CP}\n${CI}\n` +
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">pwd${CP}\n${CI}\n${YIELD}`
    )
  })
})

// =============================================================================
// CATEGORY 8: TOKEN MASK — Verify parallel paths are live
// =============================================================================

describe('greedy param-body — token mask verification', () => {
  it('after close tag, content continuation is accepted', () => {
    // Feed close tag then continue with body text — must not reject
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">hello${CP}world${CP}\n${CI}\n${YIELD}`
    )
  })

  it('after close tag, structural continuation is accepted', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">hello${CP}\n${CI}\n${YIELD}`
    )
  })

  it('after close tag + newline, content continuation is accepted', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">hello${CP}\nworld${CP}\n${CI}\n${YIELD}`
    )
  })

  it('after close tag + newline, structural continuation is accepted', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">hello${CP}\n${CI}\n${YIELD}`
    )
  })

  it('after close tag + param open prefix, content is still accepted', () => {
    // This is the key: even after seeing close + "<magnitude:parameter " (full continuation prefix),
    // the content path should still be alive with greedy grammar
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">X${CP}\n${OP} name="y">Z STILL CONTENT${CP}\n${CI}\n${YIELD}`
    )
  })
})

// =============================================================================
// CATEGORY 9: STRESS — Many embedded close tags
// =============================================================================

describe('greedy param-body — stress tests', () => {
  it('5 embedded close tags', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">` +
      `a${CP}b${CP}c${CP}d${CP}e${CP}f${CP}\n${CI}\n${YIELD}`
    )
  })

  it('10 embedded close tags', () => {
    const body = Array.from({length: 10}, (_, i) => String.fromCharCode(65 + i) + CP).join('')
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">${body}END${CP}\n${CI}\n${YIELD}`
    )
  })

  it('embedded close tags with varied separators', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">` +
      `a${CP} b${CP}\nc${CP}\t\td${CP}<div>${CP}end${CP}\n${CI}\n${YIELD}`
    )
  })
})