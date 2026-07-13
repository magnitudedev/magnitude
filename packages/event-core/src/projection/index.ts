export {
  define,
  type ProjectionInstance,
  type ProjectionConfig,
  type ProjectionResult,
  type StateOfProjection,
  type AnyProjectionResult,
  type ReadFn,
  type AmbientReader,
  type SignalSubscription
} from './define'

import * as consumer from './consumer'
import { record, sequence } from './addressed'

export const addressed = {
  record,
  sequence
}

export { consumer }

export type {
  ProjectionAddressedIndex,
  ProjectionAddressedConsumers,
  ProjectionForkedAddressedConsumers,
  ProjectionAddressedRecordHandle,
  ProjectionAddressedRecordIndex,
  ProjectionAddressedSequenceHandle
} from './addressed'

export type {
  RuntimeConsumer,
  ProjectionConsumerService,
  ProjectionRead,
  StateOf as ConsumerStateOf,
  AddressedOf as ConsumerAddressedOf
} from './consumer'

export {
  defineForked,
  type ForkableEvent,
  type ForkedState,
  type ForkedProjectionInstance,
  type ForkedProjectionSnapshot,
  type ForkedProjectionConfig,
  type ForkedProjectionResult,
  type ForkedReadFn,
  type ForkedSignalReadFn,
  type ForkedSignalHandlerBuilder,
  type ForkedAmbientHandlerBuilder,
  type ForkedSignalHandlerPair,
  type ForkedAmbientHandlerPair
} from './defineForked'

// export {
//   fsmSingleton,
//   type FSMSingletonInstance,
//   type FSMSingletonSignals,
//   type FSMSingletonSignalPubSubs,
//   type FSMSingletonEventHandler,
//   type FSMSingletonConfig,
//   type FSMSingletonResult
// } from './fsmSingleton'

// export {
//   fsmCollection,
//   type FSMCollectionState,
//   type FSMCollectionInstance,
//   type FSMCollectionSignals,
//   type FSMCollectionSignalPubSubs,
//   type FSMCollectionSpawner,
//   type FSMCollectionEventHandler,
//   type FSMCollectionConfig,
//   type FSMCollectionResult
// } from './fsmCollection'
