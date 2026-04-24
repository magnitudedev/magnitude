import type {
  AmbiguousMagnitudeCloseError,
  DuplicateParameterError,
  IncompleteToolError,
  InvalidMagnitudeOpenError,
  JsonStructuralError,
  MalformedTagError,
  MissingRequiredFieldError,
  MissingToolNameError,
  ParseErrorDetail,
  SchemaCoercionError,
  StrayCloseTagError,
  UnclosedThinkError,
  UnexpectedContentError,
  UnknownParameterError,
  UnknownToolError,
} from "../types"

export interface ErrorPresentation<E = unknown> {
  headline: (error: E) => string
  hints: (error: E) => string[]
  snippetStrategy: 'point' | 'block'
}

type ErrorCatalog = {
  [E in ParseErrorDetail as E['_tag']]: ErrorPresentation<E>
}

const asTag = (tagName: string) => `<${tagName}>`
const asCloseTag = (tagName: string) => `</${tagName}>`
const stripPrefix = (tagName: string) =>
  tagName.startsWith('magnitude:') ? tagName.slice('magnitude:'.length) : tagName

export const ERROR_CATALOG: ErrorCatalog = {
  InvalidMagnitudeOpen: {
    headline: (error) => {
      if (!error.parentTagName) {
        return `Unrecognized tag ${error.raw} at the top level.`
      }
      if (error.parentTagName === 'magnitude:invoke') {
        return `Invalid tag ${error.raw} inside tool call. Only parameters and filters are allowed here.`
      }
      return `Invalid nested tag ${error.raw} inside ${asTag(error.parentTagName)}. Magnitude tags cannot be nested.`
    },
    hints: (error) => {
      const base = 'Tags with the magnitude: prefix are ALWAYS interpreted, no matter where they appear.'
      switch (error.parentTagName) {
        case 'magnitude:reason':
          return [
            base,
            `You are not allowed to use magnitude: tags inside a reason block. Drop the prefix when reasoning about a tag — write ${asTag(stripPrefix(error.tagName))} instead of ${error.raw}.`,
          ]
        case 'magnitude:message':
          return [
            base,
            `You are not allowed to use magnitude: tags inside a message. Drop the prefix when referring to a tag — write ${asTag(stripPrefix(error.tagName))} instead of ${error.raw}.`,
          ]
        case 'magnitude:parameter':
          return [
            base,
            `You are not allowed to use magnitude: tags inside a parameter value. Drop the prefix when discussing the tag — write ${asTag(stripPrefix(error.tagName))} instead of ${error.raw}.`,
            `If you need the literal text "magnitude:" inside written code, build it from a constant (e.g. \`const PREFIX = "magnitude:"\`) and concatenate at runtime.`,
          ]
        case 'magnitude:invoke':
          return [
            'Only <magnitude:parameter name="..."> and <magnitude:filter> are allowed inside <magnitude:invoke>.',
            'Move plain text or other content outside the tool call.',
          ]
        case 'magnitude:filter':
          return [
            'A <magnitude:filter> body must be a JSONPath string only — no nested tags.',
          ]
        default:
          return [
            base,
            'Only valid magnitude tags are recognized at the top level (reason, message, invoke, yield).',
            'Check the tag name spelling.',
          ]
      }
    },
    snippetStrategy: 'point',
  },

  AmbiguousMagnitudeClose: {
    headline: (error) => {
      const expected = error.expectedTagName ? asTag(error.expectedTagName) : 'the current open tag'
      return `Mismatched close ${error.raw} while inside ${expected}.`
    },
    hints: (error) => {
      const expected = error.expectedTagName ? asCloseTag(error.expectedTagName) : 'the matching close tag'
      return [
        `Close the tag that is actually open: emit ${expected} before closing anything else.`,
        'Magnitude close tags are always interpreted — do not write them as literal text inside reasons, messages, or parameters.',
      ]
    },
    snippetStrategy: 'point',
  },

  StrayCloseTag: {
    headline: (error) => `Unexpected close ${asCloseTag(error.tagName)} with no matching open.`,
    hints: () => [
      'A close tag can only appear after its corresponding open tag.',
      'Magnitude close tags are always interpreted — drop the magnitude: prefix if you meant to refer to the tag in prose.',
    ],
    snippetStrategy: 'point',
  },

  UnknownTool: {
    headline: (error) => `Unknown tool '${error.tagName}'.`,
    hints: () => [
      'Invoke only tools that are available in this session.',
      'Check the tool name spelling.',
    ],
    snippetStrategy: 'point',
  },

  MissingToolName: {
    headline: () => 'Tool invocation is missing its tool name.',
    hints: () => [
      'Use <magnitude:invoke tool="..."> with a tool name.',
      'Or use a tool alias like <magnitude:shell> instead.',
    ],
    snippetStrategy: 'point',
  },

  UnexpectedContent: {
    headline: () => 'Unexpected content inside tool call.',
    hints: () => [
      'Move plain text outside the tool call into a message or reason block.',
      'Inside <magnitude:invoke>, wrap all input in <magnitude:parameter> or <magnitude:filter>.',
    ],
    snippetStrategy: 'point',
  },

  UnclosedThink: {
    headline: () => 'Reasoning block was left open when the response ended.',
    hints: () => [
      'Close every <magnitude:reason> block with </magnitude:reason> before yielding.',
    ],
    snippetStrategy: 'block',
  },

  MalformedTag: {
    headline: () => 'Malformed tag could not be interpreted.',
    hints: () => [
      'Ensure tag attributes are properly quoted: tool="name".',
    ],
    snippetStrategy: 'point',
  },

  UnknownParameter: {
    headline: (error) => `Unknown parameter '${error.parameterName}' for tool '${error.tagName}'.`,
    hints: () => [
      'Use only parameters defined by the tool.',
      'Check the parameter name spelling.',
    ],
    snippetStrategy: 'block',
  },

  DuplicateParameter: {
    headline: (error) => `Duplicate parameter '${error.parameterName}' for tool '${error.tagName}'.`,
    hints: () => [
      'Each parameter may only appear once per tool call.',
      'Rewrite the tool call with a single value instead of repeating the parameter.',
    ],
    snippetStrategy: 'block',
  },

  MissingRequiredField: {
    headline: (error) => `Missing required parameter '${error.parameterName}' for tool '${error.tagName}'.`,
    hints: () => [
      'Include all required parameters before closing the tool call.',
    ],
    snippetStrategy: 'block',
  },

  SchemaCoercionError: {
    headline: (error) => `Parameter '${error.parameterName}' for tool '${error.tagName}' could not be parsed as the expected type.`,
    hints: () => [
      'Format the parameter value to match the expected type.',
    ],
    snippetStrategy: 'block',
  },

  JsonStructuralError: {
    headline: (error) => `Parameter '${error.parameterName}' for tool '${error.tagName}' contains invalid structured data.`,
    hints: () => [
      'Ensure the parameter body contains valid structured data.',
      'Check that objects, arrays, quotes, and commas are balanced correctly.',
    ],
    snippetStrategy: 'block',
  },

  IncompleteTool: {
    headline: (error) => `Tool call '${error.tagName}' was not closed before the response ended.`,
    hints: () => [
      'Close the tool call with </magnitude:invoke> before yielding.',
      'Ensure all parameters and filters are closed first.',
    ],
    snippetStrategy: 'block',
  },
}
