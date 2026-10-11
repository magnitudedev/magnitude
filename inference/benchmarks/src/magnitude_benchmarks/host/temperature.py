"""Read-only Apple SMC temperature sensors; no subprocess, root, or GPU work.

The private SMC ABI and sensor-prefix convention are also used by macmon:
https://github.com/vladkens/macmon/blob/main/src_lib/sources.rs
https://github.com/vladkens/macmon/blob/main/src_lib/metrics.rs
Prefixes identify CPU/GPU-associated sensors, not documented physical core IDs.
"""

import ctypes as c
import math
import struct
import sys


class _Version(c.Structure):
    _fields_ = [
        ("major", c.c_uint8),
        ("minor", c.c_uint8),
        ("build", c.c_uint8),
        ("reserved", c.c_uint8),
        ("release", c.c_uint16),
    ]


class _Limits(c.Structure):
    _fields_ = [
        ("version", c.c_uint16),
        ("length", c.c_uint16),
        ("cpu", c.c_uint32),
        ("gpu", c.c_uint32),
        ("memory", c.c_uint32),
    ]


class _Info(c.Structure):
    _fields_ = [("size", c.c_uint32), ("type", c.c_uint32), ("attributes", c.c_uint8)]


class _Data(c.Structure):
    _fields_ = [
        ("key", c.c_uint32),
        ("version", _Version),
        ("limits", _Limits),
        ("info", _Info),
        ("result", c.c_uint8),
        ("status", c.c_uint8),
        ("command", c.c_uint8),
        ("index", c.c_uint32),
        ("data", c.c_uint8 * 32),
    ]


def sensor_group(key: str) -> str | None:
    if key.startswith(("Tp", "Te", "Ts")):
        return "cpu"
    if key.startswith("Tg"):
        return "gpu"
    return None


class AppleSMC:
    """One connection; discover float temperature keys once, cache their metadata."""

    source = "AppleSMC"

    def __init__(self):
        if sys.platform != "darwin":
            raise OSError("AppleSMC temperature readings require macOS")
        self._io = c.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
        system = c.CDLL("/usr/lib/libSystem.B.dylib")
        self._conn = c.c_uint32()
        self._keys: dict[str, _Info] = {}
        self._bind("IOServiceMatching", [c.c_char_p], c.c_void_p)
        self._bind("IOServiceGetMatchingServices", [c.c_uint32, c.c_void_p, c.POINTER(c.c_uint32)])
        self._bind("IOIteratorNext", [c.c_uint32], c.c_uint32)
        self._bind("IORegistryEntryGetName", [c.c_uint32, c.c_char_p])
        self._bind("IOObjectRelease", [c.c_uint32])
        self._bind("IOServiceOpen", [c.c_uint32, c.c_uint32, c.c_uint32, c.POINTER(c.c_uint32)])
        self._bind("IOServiceClose", [c.c_uint32])
        self._bind(
            "IOConnectCallStructMethod",
            [
                c.c_uint32,
                c.c_uint32,
                c.POINTER(_Data),
                c.c_size_t,
                c.POINTER(_Data),
                c.POINTER(c.c_size_t),
            ],
        )
        iterator = c.c_uint32()
        self._check(
            self._io.IOServiceGetMatchingServices(
                0, self._io.IOServiceMatching(b"AppleSMC"), c.byref(iterator)
            )
        )
        try:
            while service := self._io.IOIteratorNext(iterator.value):
                try:
                    name = c.create_string_buffer(128)
                    self._check(self._io.IORegistryEntryGetName(service, name))
                    if name.value == b"AppleSMCKeysEndpoint":
                        task = c.c_uint32.in_dll(system, "mach_task_self_").value
                        self._check(self._io.IOServiceOpen(service, task, 0, c.byref(self._conn)))
                        break
                finally:
                    self._io.IOObjectRelease(service)
        finally:
            self._io.IOObjectRelease(iterator.value)
        if not self._conn.value:
            raise OSError("AppleSMCKeysEndpoint is unavailable")
        try:
            info = self._read("#KEY", command=9).info
            count = int.from_bytes(bytes(self._read("#KEY", info=info).data[:4]), "big")
            if not 0 < count <= 65536:
                raise OSError(f"invalid SMC key count: {count}")
            for index in range(count):
                key = self._read(command=8, index=index).key.to_bytes(4, "big").decode("ascii")
                if sensor_group(key) is None:
                    continue
                info = self._read(key, command=9).info
                if info.size == 4 and info.type == int.from_bytes(b"flt ", "big"):
                    self._keys[key] = info
            if not self._keys:
                raise OSError("no supported CPU/GPU float temperature sensors")
        except BaseException:
            self.close()
            raise

    def _bind(self, name, args, result: type = c.c_int):
        fn = getattr(self._io, name)
        fn.argtypes, fn.restype = args, result

    @staticmethod
    def _check(code):
        if code:
            raise OSError(f"SMC IOKit call failed: 0x{code & 0xFFFFFFFF:08x}")

    def _read(self, key="", *, command=5, index=0, info=None):
        request = _Data(
            key=int.from_bytes(key.encode("ascii"), "big"), command=command, index=index
        )
        if info is not None:
            request.info = info
        response, size = _Data(), c.c_size_t(c.sizeof(_Data))
        self._check(
            self._io.IOConnectCallStructMethod(
                self._conn.value,
                2,
                c.byref(request),
                c.sizeof(request),
                c.byref(response),
                c.byref(size),
            )
        )
        if size.value != c.sizeof(_Data) or response.result:
            raise OSError(f"invalid SMC response: size={size.value}, result={response.result}")
        return response

    def read(self) -> dict:
        sensors, errors = {}, {}
        for key, info in self._keys.items():
            try:
                value = struct.unpack("<f", bytes(self._read(key, info=info).data[:4]))[0]
                if not math.isfinite(value) or not 0 < value <= 150:
                    raise ValueError(f"invalid temperature: {value}")
                sensors[key] = value
            except (OSError, ValueError) as error:
                errors[key] = str(error)
        return {"sensors_c": sensors, "errors": errors}

    def close(self):
        if self._conn.value:
            self._io.IOServiceClose(self._conn.value)
            self._conn.value = 0
