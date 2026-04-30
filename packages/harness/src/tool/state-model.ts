import type { ToolLifecycleEvent } from "../events"

// ── Phase ────────────────────────────────────────────────────────────

export type Phase = "streaming" | "executing" | "completed" | "error" | "rejected" | "interrupted"

// ── Base State ───────────────────────────────────────────────────────

export interface BaseState {
  readonly phase: Phase
}

// ── State Model ──────────────────────────────────────────────────────

export interface StateModel<
  TState extends BaseState = BaseState,
  TInput = unknown,
  TOutput = unknown,
  TEmission = unknown,
  TError = unknown,
> {
  readonly initial: TState
  readonly reduce: (state: TState, event: ToolLifecycleEvent<TInput, TOutput, TEmission, TError>) => TState
}

// ── Inference-only tool shape ────────────────────────────────────────

/**
 * Minimal shape used purely for generic type inference in defineStateModel.
 * Avoids coupling to the full HarnessTool type.
 */
interface ToolTypeCarrier<TInput, TOutput, TEmission, TError = never> {
  readonly definition: {
    readonly inputSchema: { readonly Type: TInput }
    readonly outputSchema: { readonly Type: TOutput }
  }
  readonly emissionSchema?: { readonly Type: TEmission }
  readonly errorSchema?: { readonly Type: TError }
}

// ── defineStateModel (curried) ───────────────────────────────────────

/**
 * Curried state model definition.
 *
 * First call binds the tool key (as a string literal) and tool (for type inference).
 * Second call provides the initial extra state and reducer.
 *
 * ```ts
 * const shellState = defineStateModel('shell', shellTool)({
 *   initial: { lastExitCode: null as number | null },
 *   reduce: (state, event) => { ... }
 * })
 * ```
 */
export function defineStateModel<
  TToolKey extends string,
  TInput,
  TOutput,
  TEmission,
  TError = never,
>(
  _toolKey: TToolKey,
  _tool: ToolTypeCarrier<TInput, TOutput, TEmission, TError>,
): <TExtra extends Record<string, unknown>>(config: {
  readonly initial: TExtra
  readonly reduce: (
    state: Readonly<{ readonly toolKey: TToolKey } & BaseState & TExtra>,
    event: ToolLifecycleEvent<TInput, TOutput, TEmission, TError>,
  ) => Readonly<{ readonly toolKey: TToolKey } & BaseState & TExtra>
}) => StateModel<Readonly<{ readonly toolKey: TToolKey } & BaseState & TExtra>, TInput, TOutput, TEmission, TError> {
  return <TExtra extends Record<string, unknown>>(config: {
    readonly initial: TExtra
    readonly reduce: (
      state: Readonly<{ readonly toolKey: TToolKey } & BaseState & TExtra>,
      event: ToolLifecycleEvent<TInput, TOutput, TEmission, TError>,
    ) => Readonly<{ readonly toolKey: TToolKey } & BaseState & TExtra>
  }): StateModel<Readonly<{ readonly toolKey: TToolKey } & BaseState & TExtra>, TInput, TOutput, TEmission, TError> => {
    // TS can't prove generic spread produces the intersection type.
    const initial = Object.freeze({
      toolKey: _toolKey,
      phase: "streaming" as const,
      ...config.initial,
    }) as Readonly<{ readonly toolKey: TToolKey } & BaseState & TExtra>

    return { initial, reduce: config.reduce }
  }
}
