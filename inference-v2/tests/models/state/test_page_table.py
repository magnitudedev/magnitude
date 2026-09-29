import pytest

from magnitude_engine.models.state.table import PageMap
from magnitude_engine.models.state.views import read_layer
from tests.models.state.test_pages import append, store, values


def test_layer_reads_reuse_mapping_while_horizons_and_contents_advance():
    storage = store()
    row = storage.create()
    row.reserve(16)
    append(row, 1, 3)
    first = read_layer((row,), 0)
    device = first.table.device
    assert read_layer((row,), 1).table is first.table
    for _ in range(6):
        append(row, 1, 4)
        view = read_layer((row,), 0)
        assert view.table is first.table
        assert view.table.device is device
        assert view.lengths == (row.length,)
    assert values(row) == [3] + [4] * 6
    # Unwritten reserved pages are in the table, never inside the read horizon.
    assert first.lengths == (1,)
    assert first.table.width == 4
    row.trim(2)
    truncated = read_layer((row,), 0)
    assert truncated.table is not first.table
    assert truncated.table.width == 1
    append(row, 5, 5)
    grown = read_layer((row,), 0)
    assert grown.table is not truncated.table
    assert values(row) == [3, 4] + [5] * 5
    storage.validate()
    row.close()
    storage.arena.close()


def test_branch_maps_preserve_prefix_and_rebuild_for_changed_batch_membership():
    storage = store()
    row = storage.create()
    append(row, 6, 1)
    prefix = row.checkpoint()
    original = read_layer((row,), 0)
    branch = storage.create(prefix)
    append(branch, 2, 2)
    group = read_layer((row, branch), 0)
    assert group.table.rows[0] is original.table.rows[0]
    assert group.table.rows[1] is not original.table.rows[0]
    assert group.lengths == (6, 8)
    assert read_layer((row, branch), 1).table is group.table
    reordered = read_layer((branch, row), 0)
    assert reordered.table.rows == tuple(reversed(group.table.rows))
    assert values(row) == [1] * 6
    assert values(branch) == [1] * 6 + [2] * 2
    branch.close()
    prefix.close()
    row.close()
    storage.arena.close()


@pytest.mark.parametrize("addresses", [(-1,), (4,), (True,), (1.5,)])
def test_mapping_validates_addresses_at_creation(addresses):
    with pytest.raises(ValueError, match="address"):
        PageMap(addresses, 4)


def test_relocation_invalidates_mapping_without_changing_logical_contents():
    storage = store(slab_pages=8)
    holes = storage.arena.allocate(3)
    row = storage.create()
    guards = []
    for index in range(3):
        append(row, 4, index + 1)
        if index < 2:
            guards.extend(storage.arena.allocate(1))
    before = read_layer((row,), 0)
    before_device = before.table.device.tolist()
    storage.arena.release(holes)
    assert row.compact() == 3
    after = read_layer((row,), 0)
    assert after.table is not before.table
    assert after.table.device.tolist() != before_device
    assert before.table.device.tolist() == before_device
    assert values(row) == [1] * 4 + [2] * 4 + [3] * 4
    storage.arena.release(tuple(guards))
    storage.validate()
    row.close()
    storage.arena.close()
