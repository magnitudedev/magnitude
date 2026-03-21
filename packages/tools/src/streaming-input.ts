export type ChildAcc = {
  body: string;
  complete: boolean;
  attrs: Record<string, string>;
};

export type StreamingInput<TFields, TChildren> = {
  fields: Partial<TFields>;
  body: string;
  children: {
    [K in keyof TChildren]?: Array<{
      body: string;
      complete: boolean;
      attrs: Record<string, string>;
    }>;
  };
};

// Empty streaming input for initialization
export function emptyStreamingInput<TFields, TChildren>(): StreamingInput<TFields, TChildren> {
  return { fields: {}, body: '', children: {} } as StreamingInput<TFields, TChildren>;
}
