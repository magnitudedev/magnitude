from __future__ import annotations

import importlib.metadata
import json
import string


def _installed_git_revision(distribution_name: str) -> str:
    distribution = importlib.metadata.distribution(distribution_name)
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        raise RuntimeError(f"{distribution_name} has no installed direct_url.json")
    try:
        revision = json.loads(direct_url)["vcs_info"]["commit_id"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"{distribution_name} is not installed from a pinned Git commit"
        ) from exc
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in string.hexdigits for character in revision)
    ):
        raise RuntimeError(f"{distribution_name} has an invalid Git revision")
    return revision


def runtime_identity() -> dict[str, str]:
    return {
        "omlx_version": importlib.metadata.version("omlx"),
        "omlx_revision": _installed_git_revision("omlx"),
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_revision": _installed_git_revision("mlx-lm"),
        "mlx_vlm_revision": _installed_git_revision("mlx-vlm"),
        "dflash_revision": _installed_git_revision("dflash-mlx"),
    }


def main() -> None:
    print(json.dumps(runtime_identity(), sort_keys=True))
