import { describe, it } from 'vitest'
import { shellValidator, multiToolValidator } from './helpers'

const YIELD = '<magnitude:yield_user/>'
const CP = '</magnitude:parameter>'
const CI = '</magnitude:invoke>'
const CF = '</magnitude:filter>'
const OP = '<magnitude:parameter'
const OF = '<magnitude:filter>'
const ESC_OPEN = '<magnitude:escape>'
const ESC_CLOSE = '</magnitude:escape>'

// =============================================================================
// CATEGORY 1: BASIC — Canonical valid forms
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

  it('param + filter is rejected in the current shell grammar path', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls${CP}\n${OF}$.stdout${CF}\n${CI}\n${YIELD}`
    )
  })

  it('close immediately followed by next param', () => {
    multiToolValidator().passes(
      `<magnitude:invoke tool="edit">\n${OP} name="path">a${CP}${OP} name="old">b${CP}${OP} name="new">c${CP}\n${CI}\n${YIELD}`
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
// =============================================================================

describe('greedy param-body — simple embedded close tags', () => {
  it('close tag + space + word', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo ${CP} done${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close tag + newline + plain text', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">code${CP}\nhello world${CP}\n${CI}\n${YIELD}`
    )
  })

  it('two embedded close tags', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">A${CP}B${CP}C${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close tag followed by HTML', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">code${CP}<div>more</div>${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close tag mid-sentence', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">use ${CP} to end${CP}\n${CI}\n${YIELD}`
    )
  })
})

// =============================================================================
// CATEGORY 3: STRICTNESS — <magnitude:... is always structural now
// =============================================================================

describe('greedy param-body — structural magnitude opens are rejected', () => {
  it('close + param open with name attr + more content + real close', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">START${CP}\n${OP} name="x">MIDDLE${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + param open (no whitespace) + content + real close', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">A${CP}${OP} name="y">B${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + whitespace + param open + content + real close', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">X${CP}  \n${OP} name="z">Y${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + param open + close + param open + real close (double fake)', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">a${CP}\n${OP} name="b">c${CP}\n${OP} name="d">e${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + filter open + content + real close', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">fake${CP}${OF}notreal${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + newline + filter open + content + real close', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">fake${CP}\n${OF}notreal${CP}\n${CI}\n${YIELD}`
    )
  })

  it('entire fake invoke block in content', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo ${CP}\n${CI}\n<magnitude:invoke tool="fake">\n${OP} name="x">y${CP}\n${CI}\nstill content${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + param open + close + invoke close + text + real close', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">A${CP}\n${OP} name="b">C${CP}\n${CI}\nD${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + invoke close + yield (exits everything)', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">fake${CP}\n${CI}\n${YIELD}still here${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + invoke close + think block + real close', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">fake${CP}\n${CI}\n<magnitude:think about="x">thought</magnitude:think>\nreal${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + invoke close + message block + real close', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">fake${CP}\n${CI}\n<magnitude:message to="user">hi</magnitude:message>\nreal${CP}\n${CI}\n${YIELD}`
    )
  })

  it('close + invoke close + new invoke + real close', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">fake${CP}\n${CI}\n<magnitude:invoke tool="other">\n${OP} name="x">y${CP}\n${CI}\nreal${CP}\n${CI}\n${YIELD}`
    )
  })

  it('filter close + invoke close + text + real filter close', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n${OF}data${CF}\n${CI}\nmore${CF}\n${CI}\n${YIELD}`
    )
  })

  it('filter close + invoke close + param open + real filter close', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n${OF}data${CF}${CI}${OP} name="x">fake${CF}\n${CI}\n${YIELD}`
    )
  })
})

// =============================================================================
// CATEGORY 4: REALISTIC CONTENT — Use escape blocks for literal magnitude markup
// =============================================================================

describe('greedy param-body — realistic content with close-tag-like text', () => {
  it('sed command replacing XML tags', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">sed 's/${CP}/${CF}/g' file.xml${CP}\n${CI}\n${YIELD}`
    )
  })

  it('echo command outputting XML — rejected (magnitude open in body must be escaped)', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo "${OP} name=\\"x\\">val${CP}"${CP}\n${CI}\n${YIELD}`
    )
  })

  it('grep for close tag pattern', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">grep '${CP}' src/*.ts${CP}\n${CI}\n${YIELD}`
    )
  })

  it('documentation about the XML format itself', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo "Use ${CP} to close a param and ${CI} to close an invoke"${CP}\n${CI}\n${YIELD}`
    )
  })

  it('python code with XML string — rejected (magnitude open in body must be escaped)', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">python3 -c "print('${OP} name=x>val${CP}')"${CP}\n${CI}\n${YIELD}`
    )
  })

  it('code comment mentioning close tag', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">cat << EOF\n# close with ${CP}\n# then ${CI}\nEOF${CP}\n${CI}\n${YIELD}`
    )
  })

  it('escaped magnitude parameter markup is rejected after escape removal', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo "${ESC_OPEN}${OP} name=\\"x\\">val${CP}${ESC_CLOSE}"${CP}\n${CI}\n${YIELD}`
    )
  })
})

// =============================================================================
// CATEGORY 5: FILTER BODY
// =============================================================================

describe('greedy filter-body — embedded close tags', () => {
  it('basic filter is rejected in the current shell grammar path', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls${CP}\n${OF}$.stdout${CF}\n${CI}\n${YIELD}`
    )
  })

  it('filter with embedded close is rejected in the current shell grammar path', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls${CP}\n${OF}$.stdout${CF}\nhello${CF}\n${YIELD}`
    )
  })

  it('filter close + param open (valid continuation) + content + real close', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls${CP}\n${OF}data${CF}\n${OP} name="x">fake${CF}\n${CI}\n${YIELD}`
    )
  })

  it('filter close + invoke close + content + real close', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls${CP}\n${OF}data${CF}${CI}more${CF}\n${CI}\n${YIELD}`
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
  it('think + message + invoke with embedded close', () => {
    shellValidator().rejects(
      `<magnitude:think about="test">thinking</magnitude:think>\n` +
      `<magnitude:message to="user">hello</magnitude:message>\n` +
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo ${CP} works${CP}\n${CI}\n${YIELD}`
    )
  })

  it('multiple invokes, second has embedded close', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls${CP}\n${CI}\n` +
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo ${CP} test${CP}\n${CI}\n${YIELD}`
    )
  })

  it('invoke with close-tag text + second invoke', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">fake${CP}${CI}real${CP}\n${CI}\n` +
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">pwd${CP}\n${CI}\n${YIELD}`
    )
  })
})

// =============================================================================
// CATEGORY 8: TOKEN MASK / ACCEPTANCE
// =============================================================================

describe('greedy param-body — token mask verification', () => {
  it('after close tag, content continuation is accepted', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">hello${CP}world${CP}\n${CI}\n${YIELD}`
    )
  })

  it('after close tag, structural continuation is accepted', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">hello${CP}\n${CI}\n${YIELD}`
    )
  })

  it('after close tag + newline, content continuation is accepted', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">hello${CP}\nworld${CP}\n${CI}\n${YIELD}`
    )
  })

  it('after close tag + newline, structural continuation is accepted', () => {
    shellValidator().passes(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">hello${CP}\n${CI}\n${YIELD}`
    )
  })

  it('after close tag + param open prefix, input is rejected because magnitude opens are structural', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">X${CP}\n${OP} name="y">Z STILL CONTENT${CP}\n${CI}\n${YIELD}`
    )
  })
})

// =============================================================================
// CATEGORY 9: STRESS
// =============================================================================

describe('greedy param-body — stress tests', () => {
  it('5 embedded close tags', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">` +
      `a${CP}b${CP}c${CP}d${CP}e${CP}f${CP}\n${CI}\n${YIELD}`
    )
  })

  it('10 embedded close tags', () => {
    const body = Array.from({ length: 10 }, (_, i) => String.fromCharCode(65 + i) + CP).join('')
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">${body}END${CP}\n${CI}\n${YIELD}`
    )
  })

  it('embedded close tags with varied separators', () => {
    shellValidator().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">` +
      `a${CP} b${CP}\nc${CP}\t\td${CP}<div>${CP}end${CP}\n${CI}\n${YIELD}`
    )
  })
})
