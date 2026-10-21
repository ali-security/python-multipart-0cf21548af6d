from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

from python_multipart.multipart import File, FormParser

if TYPE_CHECKING:
    from python_multipart.multipart import FileProtocol


def test_upload_dir_with_leading_slash_in_filename(tmp_path: Path) -> None:
    upload_dir = tmp_path / "upload"
    upload_dir.mkdir()

    # When the file_name provided has a leading slash, we should only use the basename.
    # This is to avoid directory traversal.
    to_upload = tmp_path / "foo.txt"

    file = File(
        bytes(to_upload),
        config={
            "UPLOAD_DIR": bytes(upload_dir),
            "UPLOAD_KEEP_FILENAME": True,
            "UPLOAD_KEEP_EXTENSIONS": True,
            "MAX_MEMORY_FILE_SIZE": 10,
        },
    )
    file.write(b"123456789012")
    assert not file.in_memory
    assert Path(upload_dir / "foo.txt").exists()
    assert Path(upload_dir / "foo.txt").read_bytes() == b"123456789012"


def test_upload_dir_with_dot_dot_segments_in_filename(tmp_path: Path) -> None:
    upload_dir = tmp_path / "upload"
    upload_dir.mkdir()

    # Relative ".." segments in the file_name must not escape UPLOAD_DIR either.
    file = File(
        b"../escaped.txt",
        config={
            "UPLOAD_DIR": bytes(upload_dir),
            "UPLOAD_KEEP_FILENAME": True,
            "UPLOAD_KEEP_EXTENSIONS": True,
            "MAX_MEMORY_FILE_SIZE": 10,
        },
    )
    file.write(b"123456789012")
    file.close()

    assert not file.in_memory
    assert file.file_name == b"../escaped.txt"
    assert file.actual_file_name == b"escaped.txt"
    assert (upload_dir / "escaped.txt").read_bytes() == b"123456789012"
    assert not (tmp_path / "escaped.txt").exists()


def test_upload_dir_with_absolute_filename_without_extensions(tmp_path: Path) -> None:
    upload_dir = tmp_path / "upload"
    upload_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()

    # Without UPLOAD_KEEP_EXTENSIONS only the base name is used, which must
    # still be stripped of any directory components.
    file = File(
        os.path.join(bytes(outside_dir), b"evil.sh"),
        config={"UPLOAD_DIR": bytes(upload_dir), "UPLOAD_KEEP_FILENAME": True, "MAX_MEMORY_FILE_SIZE": 10},
    )
    file.write(b"123456789012")
    file.close()

    assert not file.in_memory
    assert file.actual_file_name == b"evil"
    assert (upload_dir / "evil").read_bytes() == b"123456789012"
    assert not (outside_dir / "evil").exists()


def test_form_parser_keeps_uploads_inside_upload_dir(tmp_path: Path) -> None:
    upload_dir = tmp_path / "upload"
    upload_dir.mkdir()
    absolute_target = tmp_path / "absolute.txt"

    # Both a relative ("../") and an absolute filename supplied by the client in
    # the Content-Disposition header must end up inside UPLOAD_DIR.
    data = (
        b"--boundary\r\n"
        b'Content-Disposition: form-data; name="file1"; filename="../relative.txt"\r\n'
        b"Content-Type: text/plain\r\n\r\n"
        b"123456789012\r\n"
        b"--boundary\r\n"
        b'Content-Disposition: form-data; name="file2"; filename="' + bytes(absolute_target) + b'"\r\n'
        b"Content-Type: text/plain\r\n\r\n"
        b"abcdefghijkl\r\n"
        b"--boundary--\r\n"
    )

    files: list[File] = []

    def on_file(f: FileProtocol) -> None:
        files.append(cast(File, f))

    parser = FormParser(
        "multipart/form-data",
        on_field=Mock(),
        on_file=on_file,
        boundary="boundary",
        config={
            "UPLOAD_DIR": bytes(upload_dir),
            "UPLOAD_KEEP_FILENAME": True,
            "UPLOAD_KEEP_EXTENSIONS": True,
            "MAX_MEMORY_FILE_SIZE": 10,
        },
    )
    parser.write(data)
    parser.finalize()
    for f in files:
        f.close()

    assert [f.actual_file_name for f in files] == [b"relative.txt", b"absolute.txt"]
    assert (upload_dir / "relative.txt").read_bytes() == b"123456789012"
    assert (upload_dir / "absolute.txt").read_bytes() == b"abcdefghijkl"
    assert not (tmp_path / "relative.txt").exists()
    assert not absolute_target.exists()
