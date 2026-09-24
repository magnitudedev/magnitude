"""Native extraction acceptance; no installed application is modified."""
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile

HELPER = str(Path(sys.argv.pop(1)).resolve())


class Extraction(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="magnitude-extraction-")
        self.root = Path(self.temporary.name)
        self.stage = self.root / "stage"
        self.stage.mkdir(mode=0o700)
        self.archive = self.root / "update.zip"
        self.outside = self.root / "outside"
        self.outside.write_text("preserved")

    def tearDown(self):
        self.assertEqual(self.outside.read_text(), "preserved")
        self.temporary.cleanup()

    def create_archive(self, entries):
        with warnings.catch_warnings(), zipfile.ZipFile(self.archive, "w", zipfile.ZIP_DEFLATED) as archive:
            warnings.simplefilter("ignore", UserWarning)
            for name, mode, content in entries:
                entry = zipfile.ZipInfo(name)
                entry.create_system = 3
                entry.external_attr = mode << 16
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, content)

    def command(self):
        data = self.archive.read_bytes()
        return [HELPER, str(self.archive), str(self.stage), hashlib.sha256(data).hexdigest(), str(len(data))]

    def extract(self, accepted):
        result = subprocess.run(self.command(), capture_output=True, timeout=15)
        self.assertEqual(result.returncode == 0, accepted, result.stderr.decode())

    def test_authenticated_digest_and_length_are_required_before_extraction(self):
        self.create_archive([("Magnitude.app/file", stat.S_IFREG | 0o600, b"verified")])
        good = self.command()
        for digest, size in [("0" * 64, good[4]), (good[3], str(int(good[4]) + 1)),
                             (good[3], "-1"), (good[3], "0"), ("invalid", good[4])]:
            with self.subTest(digest=digest, size=size):
                result = subprocess.run(good[:3] + [digest, size], capture_output=True, timeout=15)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(list(self.stage.iterdir()), [])

    def test_archive_writers_and_hard_links_are_refused(self):
        self.create_archive([("Magnitude.app/file", stat.S_IFREG | 0o600, b"verified")])
        self.archive.chmod(0o666)
        self.extract(False)
        self.archive.chmod(0o600)
        os.link(self.archive, self.root / "alias.zip")
        self.extract(False)
        self.assertEqual(list(self.stage.iterdir()), [])

    def test_extended_staging_access_is_refused(self):
        self.create_archive([("Magnitude.app/file", stat.S_IFREG | 0o600, b"verified")])
        subprocess.run(["/bin/chmod", "+a", "everyone allow read,search", str(self.stage)], check=True)
        self.extract(False)
        self.assertEqual(list(self.stage.iterdir()), [])

    def test_regular_modes_and_framework_links(self):
        self.create_archive([
            ("Magnitude.app/Contents/MacOS/program", stat.S_IFREG | 0o755, b"program"),
            ("Magnitude.app/Contents/Frameworks/Test.framework/Versions/A/Test", stat.S_IFREG | 0o755, b"library"),
            ("Magnitude.app/Contents/Frameworks/Test.framework/Versions/Current", stat.S_IFLNK | 0o777, b"A"),
            ("Magnitude.app/Contents/Frameworks/Test.framework/Test", stat.S_IFLNK | 0o777, b"Versions/Current/Test"),
        ])
        self.extract(True)
        self.assertEqual((self.stage / "Magnitude.app/Contents/MacOS/program").stat().st_mode & 0o777, 0o755)
        link = self.stage / "Magnitude.app/Contents/Frameworks/Test.framework/Test"
        self.assertEqual(os.readlink(link), "Versions/Current/Test")
        self.assertEqual(link.read_bytes(), b"library")

    def test_invalid_paths(self):
        for path in ["../outside", str(self.outside), "Other.app/file", "Magnitude.app/../outside",
                     "Magnitude.app//file", "Magnitude.app/./file", "Magnitude.app/a\\b", "Magnitude.app/a:b"]:
            with self.subTest(path=path):
                self.create_archive([(path, stat.S_IFREG | 0o644, b"changed")])
                self.extract(False)
                self.assertEqual(list(self.stage.iterdir()), [])

    def test_invalid_link_targets(self):
        for target in [str(self.root), "../../outside", "../outside", "a/../b", "a\\b"]:
            with self.subTest(target=target):
                self.create_archive([("Magnitude.app/link", stat.S_IFLNK | 0o777, target)])
                self.extract(False)
                self.assertEqual(list(self.stage.iterdir()), [])

    def test_link_cycles(self):
        self.create_archive([("Magnitude.app/a", stat.S_IFLNK | 0o777, "b"),
                             ("Magnitude.app/b", stat.S_IFLNK | 0o777, "a")])
        self.extract(False)

    def test_write_through_link(self):
        self.create_archive([("Magnitude.app/real/", stat.S_IFDIR | 0o755, b""),
                             ("Magnitude.app/link", stat.S_IFLNK | 0o777, "real"),
                             ("Magnitude.app/link/file", stat.S_IFREG | 0o644, b"redirected")])
        self.extract(False)
        self.assertFalse((self.stage / "Magnitude.app/real/file").exists())

    def test_duplicate_entries(self):
        self.create_archive([("Magnitude.app/file", stat.S_IFREG | 0o644, b"first"),
                             ("Magnitude.app/file", stat.S_IFREG | 0o644, b"second")])
        self.extract(False)
        self.assertEqual((self.stage / "Magnitude.app/file").read_bytes(), b"first")

    def test_privilege_modes(self):
        for mode in [stat.S_IFREG | 0o4755, stat.S_IFREG | 0o2755, stat.S_IFREG | 0o1755]:
            with self.subTest(mode=mode):
                self.create_archive([("Magnitude.app/file", mode, b"")])
                self.extract(False)
                self.assertEqual(list(self.stage.iterdir()), [])

    def test_zip_special_modes_cannot_create_special_files(self):
        # The ZIP reader may normalize special attributes to regular files; it cannot create devices.
        self.create_archive([("Magnitude.app/pipe", stat.S_IFIFO | 0o600, b""),
                             ("Magnitude.app/device", stat.S_IFCHR | 0o600, b"")])
        result = subprocess.run(self.command(), capture_output=True, timeout=15)
        self.assertIn(result.returncode, [0, 1])
        for name in ["pipe", "device"]:
            path = self.stage / "Magnitude.app" / name
            if path.exists():
                self.assertTrue(stat.S_ISREG(path.lstat().st_mode))

    def test_existing_staging_is_not_modified(self):
        marker = self.stage / "retain"
        marker.write_text("retained")
        self.create_archive([("Magnitude.app/file", stat.S_IFREG | 0o644, b"content")])
        self.extract(False)
        self.assertEqual(list(self.stage.iterdir()), [marker])
        self.assertEqual(marker.read_text(), "retained")

    def test_nonprivate_staging_is_rejected(self):
        self.stage.chmod(0o755)
        self.create_archive([("Magnitude.app/file", stat.S_IFREG | 0o644, b"content")])
        self.extract(False)
        self.assertEqual(list(self.stage.iterdir()), [])

    def test_truncated_archive(self):
        self.create_archive([("Magnitude.app/file", stat.S_IFREG | 0o644, b"content")])
        self.archive.write_bytes(self.archive.read_bytes()[:-15])
        self.extract(False)

    def test_bad_content_checksum(self):
        with zipfile.ZipFile(self.archive, "w", zipfile.ZIP_STORED) as archive:
            archive.writestr("Magnitude.app/file", b"unique payload bytes")
        data = self.archive.read_bytes()
        self.archive.write_bytes(data.replace(b"unique payload bytes", b"broken payload bytes"))
        self.extract(False)

    def test_file_directory_conflict(self):
        self.create_archive([("Magnitude.app/Contents", stat.S_IFREG | 0o644, b"file"),
                             ("Magnitude.app/Contents/data", stat.S_IFREG | 0o644, b"child")])
        self.extract(False)

    def test_dangling_link(self):
        self.create_archive([("Magnitude.app/link", stat.S_IFLNK | 0o777, "missing")])
        self.extract(False)

    def test_ditto_metadata_roundtrip(self):
        bundle = self.root / "Magnitude.app"
        resources = bundle / "Contents/Resources"
        resources.mkdir(parents=True)
        resource = resources / "data"
        resource.write_bytes(b"sealed data")
        subprocess.run(["/usr/bin/xattr", "-w", "com.magnitude.extraction-fixture", "metadata", str(resource)], check=True)
        subprocess.run(["/usr/bin/ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(bundle), str(self.archive)], check=True)
        self.extract(True)
        extracted = self.stage / "Magnitude.app/Contents/Resources/data"
        self.assertEqual(extracted.read_bytes(), b"sealed data")
        value = subprocess.check_output(["/usr/bin/xattr", "-p", "com.magnitude.extraction-fixture", str(extracted)])
        self.assertEqual(value.strip(), b"metadata")


if __name__ == "__main__":
    unittest.main()
