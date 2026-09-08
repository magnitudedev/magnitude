"""Build the version-qualified MLX graph bridge at installation, never at runtime."""

import json
import os
import subprocess
import sysconfig
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        import mlx.core
        import nanobind

        root = Path(self.root)
        mlx = Path(mlx.core.__file__).parent
        nb = Path(nanobind.include_dir()).parent
        output = (
            root
            / "src/magnitude_engine/kernels/core"
            / ("_graph" + sysconfig.get_config_var("EXT_SUFFIX"))
        )
        subprocess.run(
            [
                os.environ.get("CXX", "c++"),
                "-std=c++20",
                "-O2",
                "-shared",
                "-undefined",
                "dynamic_lookup",
                "-fvisibility=hidden",
                "-DNB_DOMAIN=mlx",
                "-DMAGNITUDE_GRAPH_SOURCE=" + json.dumps((root / "native/graph.cpp").read_text()),
                "-I" + nanobind.include_dir(),
                "-I" + str(nb / "ext/robin_map/include"),
                "-I" + sysconfig.get_paths()["include"],
                "-I" + str(mlx / "include"),
                str(root / "native/graph.cpp"),
                str(nb / "src/nb_combined.cpp"),
                "-L" + str(mlx / "lib"),
                "-lmlx",
                "-Wl,-rpath,@loader_path/../../../mlx/lib",
                "-o",
                str(output),
            ],
            check=True,
        )
        build_data["pure_python"] = False
        build_data["infer_tag"] = True
        build_data["artifacts"].append(str(output.relative_to(root)))
