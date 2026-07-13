import { Component, createElement, type ReactNode } from 'react'

interface ErrorBoundaryProps {
  fallback: (error: Error) => ReactNode
  children?: ReactNode
}

interface State {
  error: Error | null
}

class ErrorBoundaryClass extends Component<ErrorBoundaryProps, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  render(): ReactNode {
    if (this.state.error) {
      return this.props.fallback(this.state.error)
    }
    return this.props.children
  }
}

// Functional wrapper to avoid opentui JSX type mismatch with React class components
export function ErrorBoundary({ fallback, children }: { fallback: (error: Error) => ReactNode, children: ReactNode }) {
  return createElement(ErrorBoundaryClass, { fallback }, children)
}
