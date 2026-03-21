export interface ToolBinding<TInput, TStreaming> {
  readonly _tool?: TInput;
  readonly _streaming?: TStreaming;
}

export type BindingStreaming<B> = B extends ToolBinding<any, infer S> ? S : never;
