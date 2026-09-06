from dataclasses import dataclass

import pytest

from magnitude_engine.engine.prefixes.index import PrefixIdentity, PrefixStore


@dataclass
class Checkpoint:
    length: int
    closed: bool = False

    def close(self):
        self.closed = True


def identity(tokens, namespace=b"target+drafter+state-v1", media=b""):
    return PrefixIdentity(
        namespace,
        tuple((token, media if index == 1 else b"") for index, token in enumerate(tokens)),
    )


def test_radix_splits_match_only_legal_boundaries_and_keep_longest_compatible_prefix():
    store = PrefixStore()
    short, long, branch = Checkpoint(2), Checkpoint(4), Checkpoint(3)
    store.retain(identity([1, 2, 3, 4]), long)
    store.retain(identity([1, 2]), short)
    store.retain(identity([1, 2, 9]), branch)
    for tokens, expected in (
        ([1, 2, 3, 4, 5], long),
        ([1, 2, 3, 8], short),
        ([1, 2, 9, 7], branch),
    ):
        lease = store.match(identity(tokens))
        assert lease is not None and lease.checkpoint is expected
        lease.close()
    # The final prompt token remains available to generate the first output.
    lease = store.match(identity([1, 2, 3, 4]))
    assert lease is not None and lease.checkpoint is short
    lease.close()
    assert store.match(identity([1, 2, 3], namespace=b"other-drafter")) is None
    assert store.match(identity([1, 2, 3], media=b"different-image")) is None
    store.close()


def test_retention_leases_prevent_eviction_and_deduplication_releases_incoming_state():
    store = PrefixStore()
    original, duplicate = Checkpoint(2), Checkpoint(2)
    key = identity([1, 2])
    store.retain(key, original)
    assert store.retain(key, duplicate) is original
    assert duplicate.closed and not original.closed
    lease = store.match(identity([1, 2, 3]))
    assert lease is not None
    assert store.eligible() == ()
    with pytest.raises(RuntimeError, match="lease"):
        store.discard((original,))
    lease.close()
    assert store.eligible() == (original,)
    store.discard((original,))
    assert original.closed and store.match(identity([1, 2, 3])) is None
    store.close()


def test_discarding_an_interior_checkpoint_keeps_descendant_state_matchable():
    store = PrefixStore()
    ancestor, descendant = Checkpoint(2), Checkpoint(4)
    store.retain(identity([1, 2]), ancestor)
    store.retain(identity([1, 2, 3, 4]), descendant)
    store.discard((ancestor,))
    assert store.match(identity([1, 2, 9])) is None
    lease = store.match(identity([1, 2, 3, 4, 5]))
    assert lease is not None and lease.checkpoint is descendant
    lease.close()
    store.close()
