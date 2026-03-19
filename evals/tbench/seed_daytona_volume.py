#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from daytona import AsyncDaytona, CreateSandboxFromImageParams, Image, VolumeMount
from daytona.common.errors import DaytonaError

DEFAULT_VOLUME_NAME = "magnitude-binaries"
DEFAULT_BINARY_PATH = "evals/tbench/bin/magnitude"
MOUNT_PATH = "/mnt/magnitude-volume"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed a Daytona volume with the Magnitude binary."
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_VOLUME_NAME,
        help=f"Daytona volume name (default: {DEFAULT_VOLUME_NAME})",
    )
    parser.add_argument(
        "--binary",
        default=DEFAULT_BINARY_PATH,
        help=f"Path to the magnitude binary (default: {DEFAULT_BINARY_PATH})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing binary for the computed hash",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def seed_volume(name: str, binary_path: Path, force: bool) -> str:
    if not binary_path.is_file():
        raise FileNotFoundError(f"Binary not found: {binary_path}")

    binary_bytes = binary_path.read_bytes()
    sha256 = sha256_file(binary_path)
    size = len(binary_bytes)

    hash_dir = f"{MOUNT_PATH}/magnitude/sha256/{sha256}"
    remote_binary_path = f"{hash_dir}/magnitude"
    remote_manifest_path = f"{hash_dir}/manifest.json"
    remote_current_path = f"{MOUNT_PATH}/magnitude/current"

    async with AsyncDaytona() as daytona:
        volume = await daytona.volume.get(name, create=True)

        # Wait for volume to become ready (it may be in pending_create state)
        poll_count = 0
        print(f"Volume '{name}' id={volume.id} state={volume.state}", flush=True)
        while volume.state != "ready":
            poll_count += 1
            if poll_count > 120:
                raise TimeoutError(f"Volume '{name}' still in state '{volume.state}' after 240s")
            print(f"  Waiting... state={volume.state} ({poll_count * 2}s elapsed)", flush=True)
            await asyncio.sleep(2)
            volume = await daytona.volume.get(name)
        print(f"Volume '{name}' is ready", flush=True)

        sandbox = None

        try:
            sandbox = await daytona.create(
                CreateSandboxFromImageParams(
                    language="python",
                    image=Image.base("ubuntu:22.04"),
                    volumes=[
                        VolumeMount(
                            volume_id=volume.id,
                            mount_path=MOUNT_PATH,
                        )
                    ],
                    auto_stop_interval=0,
                    auto_delete_interval=-1,
                )
            )

            await sandbox.fs.create_folder(f"{MOUNT_PATH}/magnitude", "755")
            await sandbox.fs.create_folder(f"{MOUNT_PATH}/magnitude/sha256", "755")
            await sandbox.fs.create_folder(hash_dir, "755")

            exists = False
            try:
                _ = await sandbox.fs.get_file_info(remote_binary_path)
                exists = True
            except DaytonaError:
                exists = False

            if exists and not force:
                print(f"Binary already present for sha256 {sha256}, updating current pointer only")
            else:
                print(f"Uploading binary ({size} bytes)...", flush=True)
                await sandbox.fs.upload_file(binary_bytes, remote_binary_path)
                print("Upload complete, setting permissions...", flush=True)
                await sandbox.process.exec(f"chmod +x {remote_binary_path}")

            manifest = {
                "sha256": sha256,
                "size": size,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await sandbox.fs.upload_file(
                json.dumps(manifest, indent=2).encode("utf-8"),
                remote_manifest_path,
            )
            await sandbox.fs.upload_file(f"{sha256}\n".encode("utf-8"), remote_current_path)
            return sha256
        finally:
            if sandbox is not None:
                try:
                    await sandbox.delete()
                except Exception as error:
                    print(f"Warning: failed to delete helper sandbox {sandbox.id}: {error}")


def main() -> None:
    args = parse_args()
    binary_path = Path(args.binary).expanduser().resolve()
    sha256 = asyncio.run(seed_volume(args.name, binary_path, args.force))
    print(f"Seeded Daytona volume '{args.name}' with magnitude sha256 {sha256}")


if __name__ == "__main__":
    main()