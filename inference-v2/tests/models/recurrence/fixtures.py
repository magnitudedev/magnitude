from contextlib import contextmanager

from magnitude_engine.models.state.recurrent import RecurrentImage, RecurrentSlot


@contextmanager
def recurrent_slots(layout, initial, budget):
    sources, destinations = [], []
    try:
        destination = RecurrentImage((layout,), len(initial), budget)
        destinations.extend(destination.acquire(index) for index in range(len(initial)))
        for values in initial:
            source = RecurrentImage((layout,), 1, budget).acquire(0)
            sources.append(source)
            source.image.write(0, values)
        slots = tuple(RecurrentSlot(row, 0) for row in sources)
        for slot, row in zip(slots, destinations, strict=True):
            slot.destination = row
        yield slots
    finally:
        for row in (*sources, *destinations):
            row.close()
