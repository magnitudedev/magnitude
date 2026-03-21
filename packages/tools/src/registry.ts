import type { ToolDisplayBinding } from './display';

/**
 * Display binding registry.
 * TContracts maps tool keys to their fully-typed ToolDisplayBinding.
 * Consumers call get<K>(toolKey) and receive the typed binding for that tool.
 * Internally, getAny supports erased lookup by dynamic string key.
 */
export interface DisplayBindingRegistry<TContracts extends Record<string, ToolDisplayBinding>> {
  get<K extends string & keyof TContracts>(toolKey: K): TContracts[K] | undefined;
  getAny(toolKey: string): ToolDisplayBinding | undefined;
  getDefault(): TContracts[keyof TContracts];
}
