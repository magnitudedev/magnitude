import React, { cloneElement, isValidElement, memo, useRef } from 'react'

import type { ReactElement, ReactNode } from 'react'

interface ElementProps {
  children?: ReactNode
  [key: string]: unknown
}

/**
 * Makes all text content within a React node tree non-selectable.
 *
 * This is important for interactive elements (buttons, clickable boxes) because
 * text inside them should not be selectable when the user clicks - it creates
 * a poor UX where text gets highlighted during interactions.
 *
 * Handles both `<text>` and `<span>` OpenTUI elements by adding `selectable={false}`.
 */
export function makeNodeTextNonSelectable(node: ReactNode): ReactNode {
  if (node === null || node === undefined || typeof node === 'boolean') return node
  if (typeof node === 'string' || typeof node === 'number') return node

  if (Array.isArray(node)) {
    return node.map((child, idx) => <React.Fragment key={idx}>{makeNodeTextNonSelectable(child)}</React.Fragment>)
  }

  if (!isValidElement(node)) return node

  const el = node as ReactElement<ElementProps>
  const type = el.type

  // Ensure text and span nodes are not selectable
  if (typeof type === 'string' && (type === 'text' || type === 'span')) {
    const nextProps = { ...el.props, selectable: false }
    const nextChildren = el.props.children ? makeNodeTextNonSelectable(el.props.children) : el.props.children
    return cloneElement(el, nextProps, nextChildren)
  }

  // Recurse into other host elements and components' children
  const nextChildren = el.props.children ? makeNodeTextNonSelectable(el.props.children) : el.props.children
  return cloneElement(el, el.props, nextChildren)
}

interface ButtonProps {
  onClick?: (e?: unknown) => void | Promise<unknown>
  onMouseOver?: () => void
  onMouseOut?: () => void
  style?: Record<string, unknown>
  children?: ReactNode
  // pass-through for box host props
  [key: string]: unknown
}

/**
 * A button component with proper click detection and non-selectable text.
 *
 * Key behavior:
 * - All nested `<text>`/`<span>` children are made `selectable={false}` via `makeNodeTextNonSelectable`
 * - Uses mouseDown/mouseUp tracking so hover or stray mouse events don't trigger clicks
 *
 * When to use:
 * - Use `Button` for standard button-like interactions (primary choice for clickable controls)
 */
export const Button = memo(function Button({ onClick, onMouseOver, onMouseOut, style, children, ...rest }: ButtonProps) {
  const nonSelectableChildren = makeNodeTextNonSelectable(children)

  // Track whether mouse down occurred on this element to implement proper click detection
  // This prevents hover from triggering clicks in some terminals
  const pressStartedRef = useRef(false)

  const handlePressStart = () => {
    pressStartedRef.current = true
  }

  const handlePressEnd = (e?: unknown) => {
    // Only trigger click if mouse down happened on this element
    if (pressStartedRef.current && onClick) {
      onClick(e)
    }
    pressStartedRef.current = false
  }

  const handlePointerLeave = () => {
    // Reset mouse down state when leaving the element
    pressStartedRef.current = false
    onMouseOut?.()
  }

  return (
    <box
      {...rest}
      style={style}
      onMouseDown={handlePressStart}
      onMouseUp={handlePressEnd}
      onMouseOver={onMouseOver}
      onMouseOut={handlePointerLeave}
    >
      {nonSelectableChildren}
    </box>
  )
})