import { Effect, Schema } from "effect"
import { Address, AddressTarget, EventEngine, Projection, type AddressRef } from "../src"

type Event =
  | { readonly kind: "created"; readonly id: string }
  | { readonly kind: "counted"; readonly id: string; readonly amount: number }

const Families = EventEngine.define<Event>({ discriminator: "kind" })

// @ts-expect-error "type" is not a common discriminator on Event
EventEngine.define<Event>({ discriminator: "type" })

const Accounts = Address.space({ name: "account", key: Schema.String })
const Sessions = Address.space({ name: "session", key: Schema.String })
const Attempts = Address.space({ name: "attempt", parent: Sessions, key: Schema.Number })

Attempts.at(Sessions.at("session"), 1)
// @ts-expect-error child addresses require the exact parent space
Attempts.at(Accounts.at("account"), 1)

const Counts = Families.Projection.partitioned({
  name: "Counts",
  by: Accounts,
  state: Schema.Number,
  initial: () => 0,
  handlers: (on) => [
    on.event("counted", {
      route: (event) => {
        const amount: number = event.amount
        void amount
        return AddressTarget.one(Accounts.at(event.id))
      },
    }, ({ event, state }) => Projection.set(state + event.amount)),
    // @ts-expect-error a route cannot feed addresses from another space
    on.event("created", { route: (event) => AddressTarget.one(Sessions.at(event.id)) }, () => Projection.keep),
  ],
})

declare const reference: AddressRef
// @ts-expect-error heterogeneous address references deliberately expose no erased key
reference.key

declare const reader: import("../src").ProjectionReader
const validAccountRead = reader.at(Counts, Accounts.at("account"))
// @ts-expect-error projection reads require an address from the exact declared space
const invalidSessionRead = reader.at(Counts, Sessions.at("session"))
void validAccountRead
void invalidSessionRead

type OtherEvent = { readonly kind: "other" }
const OtherFamily = EventEngine.define<OtherEvent>({ discriminator: "kind" })
const OtherProjection = OtherFamily.Projection.singleton({
  name: "Other",
  state: Schema.Number,
  initial: 0,
  handlers: () => [],
})
// @ts-expect-error components from a different event domain cannot be composed
Families.make({ name: "mixed", projections: [OtherProjection], workers: [] })

const AccountsFacade = Families.Facade.define({
  name: "Accounts",
  mutations: ({ emit, read, interrupt }) => ({
    add: (id: string, amount: number) => Effect.gen(function* () {
      yield* emit({ kind: "counted", id, amount })
      yield* interrupt(Accounts.at(id))
      return yield* read.at(Counts, Accounts.at(id))
    }),
  }),
  queries: (context) => {
    // @ts-expect-error queries have no event-emission capability
    context.emit
    // @ts-expect-error queries have no worker-interruption capability
    context.interrupt
    return {
      count: (id: string) => context.read.at(Counts, Accounts.at(id)),
    }
  },
})

declare const runtime: import("../src").EngineRuntime<Event>
const accounts = AccountsFacade.make({ runtime })
const addAccount = accounts.mutations.add("account", 1)
const getAccount = accounts.queries.count.get("account")
const accountChanges = accounts.queries.count.changes("account")
void addAccount
void getAccount
void accountChanges

const CountChanged = Families.Signal.define<{ readonly id: string; readonly count: number }>({
  name: "count-changed",
})

const SignalSource = Families.Projection.singleton({
  name: "SignalSourceTypes",
  state: Schema.Number,
  initial: 0,
  emits: [CountChanged],
  handlers: (on) => [
    on.event("counted", ({ event, emit }) => {
      emit(CountChanged, { id: event.id, count: event.amount })
      // @ts-expect-error signal payloads are checked against their definition
      emit(CountChanged, { id: event.id, value: event.amount })
      return Projection.set(event.amount)
    }),
  ],
})

const SignalConsumer = Families.Projection.partitioned({
  name: "SignalConsumerTypes",
  by: Accounts,
  state: Schema.Number,
  initial: () => 0,
  handlers: (on) => [
    on.signal(CountChanged, {
      route: (signal) => AddressTarget.one(Accounts.at(signal.payload.id)),
    }, ({ signal }) => Projection.set(signal.payload.count)),
    on.signal(CountChanged, {
      // @ts-expect-error signal routes must target the consuming projection's exact space
      route: (signal) => AddressTarget.one(Sessions.at(signal.payload.id)),
    }, () => Projection.keep),
  ],
})

const SignalWorker = Families.Worker.define({
  name: "SignalWorkerTypes",
  handlers: (on) => [
    on.signal(CountChanged, ({ signal }) => Effect.sync(() => {
      const count: number = signal.payload.count
      void count
    })),
  ],
})

Families.make({
  name: "signals-typed",
  projections: [SignalConsumer, SignalSource],
  workers: [SignalWorker],
})

// A worker handler's emit takes any event of its engine, not only the variant it handles.
Families.Worker.define({
  name: "Emitter",
  handlers: (on) => [
    on.event("created", ({ event, emit }) => emit({ kind: "counted", id: event.id, amount: 1 })),
  ],
})


