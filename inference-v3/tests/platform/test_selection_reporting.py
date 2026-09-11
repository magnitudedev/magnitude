from performance.selection import realized


class OpaqueRuntimeHandle:
    def __getattribute__(self, name):
        if name == "__dict__":
            raise RuntimeError("opaque runtime state")
        return super().__getattribute__(name)


class Component:
    def __init__(self):
        self.compiled = OpaqueRuntimeHandle()


def test_selection_reporting_treats_opaque_runtime_handles_as_leaves():
    assert realized(Component()) == ()
