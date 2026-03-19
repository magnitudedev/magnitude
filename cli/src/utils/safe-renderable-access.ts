export interface SafeRenderableOptions<T> {
  mountedRef?: React.RefObject<boolean>
  fallback: T
  onError?: (error: unknown) => void
}

export function safeRenderableAccess<R, T>(
  renderable: R | null | undefined,
  read: (renderable: R) => T,
  options: SafeRenderableOptions<T>,
): T {
  const { mountedRef, fallback, onError } = options

  if (mountedRef?.current === false) return fallback
  if (renderable == null) return fallback

  try {
    return read(renderable)
  } catch (error) {
    onError?.(error)
    return fallback
  }
}

export interface SafeRenderableCallOptions {
  mountedRef?: React.RefObject<boolean>
  onError?: (error: unknown) => void
}

export function safeRenderableCall<R>(
  renderable: R | null | undefined,
  fn: (renderable: R) => void,
  options?: SafeRenderableCallOptions,
): boolean {
  if (options?.mountedRef?.current === false) return false
  if (renderable == null) return false

  try {
    fn(renderable)
    return true
  } catch (error) {
    options?.onError?.(error)
    return false
  }
}