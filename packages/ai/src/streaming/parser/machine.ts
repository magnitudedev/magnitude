/**
 * Generic Op-based stack machine for frame-dispatch parsers.
 * Subset of xml-act's machine, trimmed to ops used by the JSON parser.
 */

export type Op<F, E> =
  | { readonly type: 'push'; readonly frame: F }
  | { readonly type: 'pop' }
  | { readonly type: 'replace'; readonly frame: F }
  | { readonly type: 'emit'; readonly event: E }

export interface StackMachine<F, E> {
  apply(ops: ReadonlyArray<Op<F, E>>): void
  peek(): F | undefined
  readonly stack: ReadonlyArray<F>
}

export function createStackMachine<F extends { readonly type: string }, E>(
  initialFrame: F,
  emit: (event: E) => void,
): StackMachine<F, E> {
  const stack: F[] = [initialFrame]

  return {
    apply(ops) {
      for (const op of ops) {
        switch (op.type) {
          case 'push':
            stack.push(op.frame)
            break
          case 'pop':
            if (stack.length > 1) stack.pop()
            break
          case 'replace':
            if (stack.length > 0) stack[stack.length - 1] = op.frame
            break
          case 'emit':
            emit(op.event)
            break
        }
      }
    },
    peek: () => stack[stack.length - 1],
    get stack() {
      return stack
    },
  }
}