/**
 * No-op Handlers
 *
 * Explicit no-ops for tokens we don't handle yet or intentionally ignore.
 * Each no-op should have a comment explaining why it's a no-op.
 */

import { definePartialHandlers } from './define'

// =============================================================================
// ENTER HANDLERS
// =============================================================================

export const enter = definePartialHandlers({
  // ---------------------------------------------------------------------------
  // GFM Autolink Literal - Handled as raw text in inline.ts
  // These tokens represent bare URLs that are auto-linked. We preserve them
  // as raw text for lossless round-trip.
  // ---------------------------------------------------------------------------
  literalAutolinkEmail: () => {},
  literalAutolinkHttp: () => {},
  literalAutolinkWww: () => {},

  // ---------------------------------------------------------------------------
  // GFM Footnotes - NOT YET IMPLEMENTED
  // These tokens are produced by micromark-extension-gfm-footnote but we
  // don't have AST nodes for footnotes yet.
  // TODO: Implement footnote support
  // ---------------------------------------------------------------------------
  gfmFootnoteCall: () => {},
  gfmFootnoteCallLabelMarker: () => {},
  gfmFootnoteCallMarker: () => {},
  gfmFootnoteCallString: () => {},
  gfmFootnoteDefinition: () => {},
  gfmFootnoteDefinitionIndent: () => {},
  gfmFootnoteDefinitionLabel: () => {},
  gfmFootnoteDefinitionLabelMarker: () => {},
  gfmFootnoteDefinitionLabelString: () => {},
  gfmFootnoteDefinitionMarker: () => {},
  gfmFootnoteDefinitionWhitespace: () => {},

  // ---------------------------------------------------------------------------
  // Math Extension - DISABLED
  // These tokens are produced by micromark-extension-math but math support
  // is currently disabled. The extension is not loaded in tokenizer.ts.
  // ---------------------------------------------------------------------------
  // mathFlow: () => {},
  // mathFlowFence: () => {},
  // mathFlowFenceMeta: () => {},
  // mathFlowFenceSequence: () => {},
  // mathFlowValue: () => {},
  // mathText: () => {},
  // mathTextData: () => {},
  // mathTextPadding: () => {},
  // mathTextSequence: () => {},
})

// =============================================================================
// EXIT HANDLERS
// =============================================================================

export const exit = definePartialHandlers({
  // ---------------------------------------------------------------------------
  // GFM Autolink Literal - Handled as raw text in inline.ts
  // ---------------------------------------------------------------------------
  literalAutolinkEmail: () => {},
  literalAutolinkHttp: () => {},
  literalAutolinkWww: () => {},

  // ---------------------------------------------------------------------------
  // GFM Footnotes - NOT YET IMPLEMENTED
  // ---------------------------------------------------------------------------
  gfmFootnoteCall: () => {},
  gfmFootnoteCallLabelMarker: () => {},
  gfmFootnoteCallMarker: () => {},
  gfmFootnoteCallString: () => {},
  gfmFootnoteDefinition: () => {},
  gfmFootnoteDefinitionIndent: () => {},
  gfmFootnoteDefinitionLabel: () => {},
  gfmFootnoteDefinitionLabelMarker: () => {},
  gfmFootnoteDefinitionLabelString: () => {},
  gfmFootnoteDefinitionMarker: () => {},
  gfmFootnoteDefinitionWhitespace: () => {},

  // ---------------------------------------------------------------------------
  // Math Extension - DISABLED
  // ---------------------------------------------------------------------------
  // mathFlow: () => {},
  // mathFlowFence: () => {},
  // mathFlowFenceMeta: () => {},
  // mathFlowFenceSequence: () => {},
  // mathFlowValue: () => {},
  // mathText: () => {},
  // mathTextData: () => {},
  // mathTextPadding: () => {},
  // mathTextSequence: () => {},
})
