
export type Op<F, E> =
  | { readonly type: 'push'; readonly frame: F }
  | { readonly type: 'pop' }
  | { readonly type: 'popUntil'; readonly predicate: (frame: F) => boolean }
  | { readonly type: 'replace'; readonly frame: F }
  | { readonly type: 'emit'; readonly event: E }
  | { readonly type: 'done' }

export interface StackMachine<F, E> {
  apply(ops: ReadonlyArray<Op<F, E>>): void
  peek(): F | undefined
  readonly stack: ReadonlyArray<F>
  readonly done: boolean
}

export function createStackMachine<F extends { readonly type: string }, E>(
  initialFrame: F,
  emit: (event: E) => void,
): StackMachine<F, E> {
  const stack: F[] = [initialFrame]
  let isDone = false

  return {
    apply(ops) {
      if (isDone) return
      for (const op of ops) {
        switch (op.type) {
          case 'push':
            stack.push(op.frame)
            break
          case 'pop':
            if (stack.length > 1) stack.pop()
            break
          case 'popUntil':
            while (stack.length > 1 && !op.predicate(stack[stack.length - 1])) {
              stack.pop()
            }
            break
          case 'replace':
            if (stack.length > 0) stack[stack.length - 1] = op.frame
            break
          case 'emit':
            emit(op.event)
            break
          case 'done':
            isDone = true
            return
        }
      }
    },
    peek: () => stack[stack.length - 1],
    get stack() {
      return stack
    },
    get done() {
      return isDone
    },
  }
}
