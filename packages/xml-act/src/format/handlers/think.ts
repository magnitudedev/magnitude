import { emit, pop, push, replace } from '../ops'
import { appendTopProse, endTopProse } from '../prose'
import { rawCloseTag, rawOpenTag } from '../raw'
import type { Resolve, TagHandler, XmlActFrame, XmlActEvent } from '../types'
import { PASSTHROUGH } from '../types'
import { findFrame } from '../types'

export function thinkHandler(
  lensesTag = 'lenses',
  makeResolve: (tagName: string, isLenses: boolean) => Resolve = () => PASSTHROUGH,
): TagHandler<XmlActFrame, XmlActEvent> {
  return {
    open(ctx) {
      const raw = rawOpenTag(ctx.tagName, ctx.attrs)
      if (!ctx.afterNewline) {
        const currentThink = findFrame(ctx.stack, 'think')
        if (currentThink && currentThink.tag === ctx.tagName) {
          if (currentThink.activeLens) {
            return [
              replace({ ...currentThink, activeLens: { ...currentThink.activeLens, body: currentThink.activeLens.body + raw } }),
              emit({ _tag: 'LensChunk', text: raw }),
            ]
          }
          return [
            replace({ ...currentThink, body: currentThink.body + raw }),
            emit({ _tag: 'ProseChunk', patternId: 'think', text: raw }),
          ]
        }
        return appendTopProse(ctx.stack, raw)
      }
      const isLenses = ctx.tagName === lensesTag
      const currentThink = findFrame(ctx.stack, 'think')
      if (currentThink && currentThink.tag === ctx.tagName) {
        return [replace({ ...currentThink, depth: currentThink.depth + 1, body: currentThink.body + raw }), emit({ _tag: 'ProseChunk', patternId: 'think', text: raw })]
      }
      return [
        ...endTopProse(ctx.stack),
        push({
          type: 'think',
          tag: ctx.tagName,
          body: '',
          depth: 0,
          about: isLenses ? null : (ctx.attrs.get('about') ?? null),
          isLenses,
          activeLens: null,
          lenses: [],
          resolve: makeResolve(ctx.tagName, isLenses),
        }),
      ]
    },
    close(ctx) {
      const think = findFrame(ctx.stack, 'think')
      if (!think) return []
      const rawClose = rawCloseTag(ctx.tagName)
      if (!ctx.afterNewline) {
        if (think.activeLens) {
          return [
            replace({ ...think, activeLens: { ...think.activeLens, body: think.activeLens.body + rawClose } }),
            emit({ _tag: 'LensChunk', text: rawClose }),
          ]
        }
        return [replace({ ...think, body: think.body + rawClose }), emit({ _tag: 'ProseChunk', patternId: 'think', text: rawClose })]
      }
      if (think.depth > 0) {
        if (think.activeLens) {
          return [
            replace({ ...think, depth: think.depth - 1, activeLens: { ...think.activeLens, body: think.activeLens.body + rawClose } }),
            emit({ _tag: 'LensChunk', text: rawClose }),
          ]
        }
        return [replace({ ...think, depth: think.depth - 1, body: think.body + rawClose }), emit({ _tag: 'ProseChunk', patternId: 'think', text: rawClose })]
      }
      if (think.isLenses) {
        if (think.activeLens) {
          return [
            emit({ _tag: 'LensEnd', name: think.activeLens.name, content: think.activeLens.body }),
            pop,
          ]
        }
        return [pop]
      }
      return [
        emit({ _tag: 'ProseEnd', patternId: 'think', content: think.body, about: think.about }),
        pop,
      ]
    },
    selfClose() {
      return []
    },
  }
}

export function lensHandler(): TagHandler<XmlActFrame, XmlActEvent> {
  return {
    open(ctx) {
      const think = findFrame(ctx.stack, 'think')
      if (!think?.isLenses) return []
      const nextName = ctx.attrs.get('name') ?? ''
      if (think.activeLens) {
        const raw = rawOpenTag('lens', ctx.attrs)
        return [
          replace({
            ...think,
            activeLens: {
              ...think.activeLens,
              body: think.activeLens.body + raw,
              depth: think.activeLens.depth + 1,
            },
          }),
          emit({ _tag: 'LensChunk', text: raw }),
        ]
      }
      return [
        emit({ _tag: 'LensStart', name: nextName }),
        replace({ ...think, activeLens: { name: nextName, body: '', depth: 0 } }),
      ]
    },
    close(ctx) {
      const think = findFrame(ctx.stack, 'think')
      if (!think?.isLenses || !think.activeLens) return []
      if (think.activeLens.depth > 0) {
        const raw = rawCloseTag('lens')
        return [
          replace({
            ...think,
            activeLens: {
              ...think.activeLens,
              body: think.activeLens.body + raw,
              depth: think.activeLens.depth - 1,
            },
          }),
          emit({ _tag: 'LensChunk', text: raw }),
        ]
      }
      const content = think.activeLens.body.replace(/^\n+|\n+$/g, '')
      return [
        emit({ _tag: 'LensEnd', name: think.activeLens.name, content }),
        replace({
          ...think,
          activeLens: null,
          lenses: [...think.lenses, { name: think.activeLens.name, body: content }],
        }),
      ]
    },
    selfClose(ctx) {
      const name = ctx.attrs.get('name') ?? ''
      return [emit({ _tag: 'LensStart', name }), emit({ _tag: 'LensEnd', name, content: '' })]
    },
  }
}