"""The local hosts file: only our choices per ssh alias: pools, cache root, model store and
host environment (`env`, applied to every step on that host, e.g. a toolkit location).
A host entry may omit `root` or `models` to use the defaults.

`hosts.json` is machine-specific and gitignored; `hosts.example.json` shows its shape.
Without it, every host gets the built-in defaults and no pools exist. Connection details
come from the user's ssh configuration. OS and architecture are probed on connect, never
stored.
"""

import json
from dataclasses import dataclass
from pathlib import Path

HOSTS_FILE = Path(__file__).resolve().parents[2] / "hosts.json"
DEFAULT_ROOT = "~/.cache/magnitude-evaluation"
DEFAULT_MODELS = "~/.magnitude/models"


@dataclass(frozen=True)
class HostConfig:
    alias: str
    root: str
    models: str
    env: tuple[tuple[str, str], ...] = ()  # applied to every step on the host; `~` allowed


@dataclass(frozen=True)
class Hosts:
    defaults: HostConfig
    pools: dict[str, tuple[str, ...]]
    hosts: dict[str, HostConfig]

    @classmethod
    def load(cls, path: Path = HOSTS_FILE) -> "Hosts":
        if not path.is_file():
            return cls(HostConfig("", DEFAULT_ROOT, DEFAULT_MODELS), {}, {})
        data = json.loads(path.read_text())
        defaults = data.get("defaults", {"root": DEFAULT_ROOT, "models": DEFAULT_MODELS})
        hosts = {
            alias: HostConfig(
                alias,
                entry.get("root", defaults["root"]),
                entry.get("models", defaults["models"]),
                tuple(entry.get("env", {}).items()),
            )
            for alias, entry in data.get("hosts", {}).items()
        }
        pools = {name: tuple(members) for name, members in data.get("pools", {}).items()}
        overlap = pools.keys() & hosts.keys()
        if overlap:
            raise ValueError(f"names are both a pool and a host: {sorted(overlap)}")
        return cls(HostConfig("", defaults["root"], defaults["models"]), pools, hosts)

    def host(self, alias: str) -> HostConfig:
        if alias in self.hosts:
            return self.hosts[alias]
        return HostConfig(alias, self.defaults.root, self.defaults.models)

    def resolve(self, name: str) -> tuple[HostConfig, ...]:
        """A pool's members, or the single named host."""
        if name in self.pools:
            return tuple(self.host(alias) for alias in self.pools[name])
        return (self.host(name),)

    def known(self) -> tuple[str, ...]:
        """Every listed host alias, pool members included, in file order."""
        aliases = dict.fromkeys(self.hosts)
        for members in self.pools.values():
            aliases.update(dict.fromkeys(members))
        return tuple(aliases)
