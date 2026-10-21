from __future__ import annotations

import os
import random
import sys
import tempfile
import time
import unittest
from io import BytesIO
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import yaml

from python_multipart.decoders import Base64Decoder, QuotedPrintableDecoder
from python_multipart.exceptions import (
    DecodeError,
    FileError,
    FormParserError,
    MultipartParseError,
    QuerystringParseError,
)
from python_multipart.multipart import (
    BaseParser,
    Field,
    File,
    FormParser,
    MultipartParser,
    OctetStreamParser,
    QuerystringParser,
    create_form_parser,
    parse_form,
    parse_options_header,
)

from .compat import parametrize, parametrize_class

if TYPE_CHECKING:
    from typing import Any, Iterator, TypedDict

    from python_multipart.multipart import FieldProtocol, FileConfig, FileProtocol

    class TestParams(TypedDict):
        name: str
        test: bytes
        result: Any


# Get the current directory for our later test cases.
curr_dir = os.path.abspath(os.path.dirname(__file__))


def force_bytes(val: str | bytes) -> bytes:
    if isinstance(val, str):
        val = val.encode(sys.getfilesystemencoding())

    return val


class TestField(unittest.TestCase):
    def setUp(self) -> None:
        self.f = Field(b"foo")

    def test_name(self) -> None:
        self.assertEqual(self.f.field_name, b"foo")

    def test_data(self) -> None:
        self.f.write(b"test123")
        self.assertEqual(self.f.value, b"test123")

    def test_cache_expiration(self) -> None:
        self.f.write(b"test")
        self.assertEqual(self.f.value, b"test")
        self.f.write(b"123")
        self.assertEqual(self.f.value, b"test123")

    def test_finalize(self) -> None:
        self.f.write(b"test123")
        self.f.finalize()
        self.assertEqual(self.f.value, b"test123")

    def test_close(self) -> None:
        self.f.write(b"test123")
        self.f.close()
        self.assertEqual(self.f.value, b"test123")

    def test_from_value(self) -> None:
        f = Field.from_value(b"name", b"value")
        self.assertEqual(f.field_name, b"name")
        self.assertEqual(f.value, b"value")

        f2 = Field.from_value(b"name", None)
        self.assertEqual(f2.value, None)

    def test_equality(self) -> None:
        f1 = Field.from_value(b"name", b"value")
        f2 = Field.from_value(b"name", b"value")

        self.assertEqual(f1, f2)

    def test_equality_with_other(self) -> None:
        f = Field.from_value(b"foo", b"bar")
        self.assertFalse(f == b"foo")
        self.assertFalse(b"foo" == f)

    def test_set_none(self) -> None:
        f = Field(b"foo")
        self.assertEqual(f.value, b"")

        f.set_none()
        self.assertEqual(f.value, None)


class TestFile(unittest.TestCase):
    def setUp(self) -> None:
        self.c: FileConfig = {}
        self.d = force_bytes(tempfile.mkdtemp())
        self.f = File(b"foo.txt", config=self.c)

    def assert_data(self, data: bytes) -> None:
        f = self.f.file_object
        f.seek(0)
        self.assertEqual(f.read(), data)
        f.seek(0)
        f.truncate()

    def assert_exists(self) -> None:
        assert self.f.actual_file_name is not None
        full_path = os.path.join(self.d, self.f.actual_file_name)
        self.assertTrue(os.path.exists(full_path))

    def test_simple(self) -> None:
        self.f.write(b"foobar")
        self.assert_data(b"foobar")

    def test_invalid_write(self) -> None:
        m = Mock()
        m.write.return_value = 5
        self.f._fileobj = m
        v = self.f.write(b"foobar")
        self.assertEqual(v, 5)

    def test_file_fallback(self) -> None:
        self.c["MAX_MEMORY_FILE_SIZE"] = 1

        self.f.write(b"1")
        self.assertTrue(self.f.in_memory)
        self.assert_data(b"1")

        self.f.write(b"123")
        self.assertFalse(self.f.in_memory)
        self.assert_data(b"123")

        # Test flushing too.
        old_obj = self.f.file_object
        self.f.flush_to_disk()
        self.assertFalse(self.f.in_memory)
        self.assertIs(self.f.file_object, old_obj)

    def test_file_fallback_with_data(self) -> None:
        self.c["MAX_MEMORY_FILE_SIZE"] = 10

        self.f.write(b"1" * 10)
        self.assertTrue(self.f.in_memory)

        self.f.write(b"2" * 10)
        self.assertFalse(self.f.in_memory)

        self.assert_data(b"11111111112222222222")

    def test_file_name(self) -> None:
        # Write to this dir.
        self.c["UPLOAD_DIR"] = self.d
        self.c["MAX_MEMORY_FILE_SIZE"] = 10

        # Write.
        self.f.write(b"12345678901")
        self.assertFalse(self.f.in_memory)

        # Assert that the file exists
        self.assertIsNotNone(self.f.actual_file_name)
        self.assert_exists()

    def test_file_full_name(self) -> None:
        # Write to this dir.
        self.c["UPLOAD_DIR"] = self.d
        self.c["UPLOAD_KEEP_FILENAME"] = True
        self.c["MAX_MEMORY_FILE_SIZE"] = 10

        # Write.
        self.f.write(b"12345678901")
        self.assertFalse(self.f.in_memory)

        # Assert that the file exists
        self.assertEqual(self.f.actual_file_name, b"foo")
        self.assert_exists()

    def test_file_full_name_with_ext(self) -> None:
        self.c["UPLOAD_DIR"] = self.d
        self.c["UPLOAD_KEEP_FILENAME"] = True
        self.c["UPLOAD_KEEP_EXTENSIONS"] = True
        self.c["MAX_MEMORY_FILE_SIZE"] = 10

        # Write.
        self.f.write(b"12345678901")
        self.assertFalse(self.f.in_memory)

        # Assert that the file exists
        self.assertEqual(self.f.actual_file_name, b"foo.txt")
        self.assert_exists()

    def test_no_dir_with_extension(self) -> None:
        self.c["UPLOAD_KEEP_EXTENSIONS"] = True
        self.c["MAX_MEMORY_FILE_SIZE"] = 10

        # Write.
        self.f.write(b"12345678901")
        self.assertFalse(self.f.in_memory)

        # Assert that the file exists
        assert self.f.actual_file_name is not None
        ext = os.path.splitext(self.f.actual_file_name)[1]
        self.assertEqual(ext, b".txt")
        self.assert_exists()

    def test_invalid_dir_with_name(self) -> None:
        # Write to this dir.
        self.c["UPLOAD_DIR"] = force_bytes(os.path.join("/", "tmp", "notexisting"))
        self.c["UPLOAD_KEEP_FILENAME"] = True
        self.c["MAX_MEMORY_FILE_SIZE"] = 5

        # Write.
        with self.assertRaises(FileError):
            self.f.write(b"1234567890")

    def test_invalid_dir_no_name(self) -> None:
        # Write to this dir.
        self.c["UPLOAD_DIR"] = force_bytes(os.path.join("/", "tmp", "notexisting"))
        self.c["UPLOAD_KEEP_FILENAME"] = False
        self.c["MAX_MEMORY_FILE_SIZE"] = 5

        # Write.
        with self.assertRaises(FileError):
            self.f.write(b"1234567890")

    # TODO: test uploading two files with the same name.


class TestParseOptionsHeader(unittest.TestCase):
    def test_simple(self) -> None:
        t, p = parse_options_header("application/json")
        self.assertEqual(t, b"application/json")
        self.assertEqual(p, {})

    def test_blank(self) -> None:
        t, p = parse_options_header("")
        self.assertEqual(t, b"")
        self.assertEqual(p, {})

    def test_single_param(self) -> None:
        t, p = parse_options_header("application/json;par=val")
        self.assertEqual(t, b"application/json")
        self.assertEqual(p, {b"par": b"val"})

    def test_single_param_with_spaces(self) -> None:
        t, p = parse_options_header(b"application/json;     par=val")
        self.assertEqual(t, b"application/json")
        self.assertEqual(p, {b"par": b"val"})

    def test_multiple_params(self) -> None:
        t, p = parse_options_header(b"application/json;par=val;asdf=foo")
        self.assertEqual(t, b"application/json")
        self.assertEqual(p, {b"par": b"val", b"asdf": b"foo"})

    def test_quoted_param(self) -> None:
        t, p = parse_options_header(b'application/json;param="quoted"')
        self.assertEqual(t, b"application/json")
        self.assertEqual(p, {b"param": b"quoted"})

    def test_quoted_param_with_semicolon(self) -> None:
        t, p = parse_options_header(b'application/json;param="quoted;with;semicolons"')
        self.assertEqual(p[b"param"], b"quoted;with;semicolons")

    def test_quoted_param_with_escapes(self) -> None:
        t, p = parse_options_header(b'application/json;param="This \\" is \\" a \\" quote"')
        self.assertEqual(p[b"param"], b'This " is " a " quote')

    def test_handles_ie6_bug(self) -> None:
        t, p = parse_options_header(b'text/plain; filename="C:\\this\\is\\a\\path\\file.txt"')

        self.assertEqual(p[b"filename"], b"file.txt")

    def test_redos_attack_header(self) -> None:
        t, p = parse_options_header(
            b'application/x-www-form-urlencoded; !="'
            b"\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
        )
        # If vulnerable, this test wouldn't finish, the line above would hang
        self.assertIn(b'"\\', p[b"!"])

    def test_ignores_rfc_2231_extended_param(self) -> None:
        # RFC 7578 §4.2 forbids the RFC 5987/2231 extended syntax, so the
        # decoded `param*` value is not exposed under `param`.
        t, p = parse_options_header(b"text/plain; param*=us-ascii'en-us'encoded%20message")

        self.assertEqual(t, b"text/plain")
        self.assertEqual(p, {})

    def test_plain_param_authoritative_over_extended(self) -> None:
        # When both plain and extended forms are present, the plain one wins
        # and the extended one is ignored.
        _, p = parse_options_header(b"form-data; name=\"comment\"; name*=utf-8''other")

        self.assertEqual(p, {b"name": b"comment"})

    def test_plain_filename_authoritative_over_extended(self) -> None:
        _, p = parse_options_header(b"form-data; name=\"file\"; filename=\"safe.txt\"; filename*=UTF-8''evil.php")

        self.assertEqual(p, {b"name": b"file", b"filename": b"safe.txt"})

    def test_ignores_rfc_2231_continuation_filename(self) -> None:
        _, p = parse_options_header(b'form-data; name="f"; filename*0="a"; filename*1="b.txt"')

        self.assertEqual(p, {b"name": b"f"})

    def test_ignores_oversized_rfc_2231_index(self) -> None:
        t, p = parse_options_header("text/plain; filename*" + ("1" * 4301) + "*=utf-8''x")

        self.assertEqual(t, b"text/plain")
        self.assertEqual(p, {})

    def test_ignores_mixed_rfc_2231_continuations(self) -> None:
        t, p = parse_options_header("text/plain; filename*=utf-8''a; filename*0*=utf-8''b")

        self.assertEqual(t, b"text/plain")
        self.assertEqual(p, {})

    def test_ignores_extended_param_case_insensitively(self) -> None:
        _, p = parse_options_header(b"text/plain; UPPER*=utf-8''X")

        self.assertEqual(p, {})

    def test_preserves_quoted_semicolons_and_escapes(self) -> None:
        _, p = parse_options_header(b'text/plain; a="x;y"; b="esc \\" quote"')

        self.assertEqual(p, {b"a": b"x;y", b"b": b'esc " quote'})

    def test_preserves_content_type_case(self) -> None:
        t, p = parse_options_header(b"Text/Plain; a=b")

        self.assertEqual(t, b"Text/Plain")
        self.assertEqual(p, {b"a": b"b"})

    def test_preserves_backslash_unquoting_order(self) -> None:
        _, p = parse_options_header(b'text/plain; q="a\\\\b"')

        self.assertEqual(p, {b"q": b"a\\b"})


class TestBaseParser(unittest.TestCase):
    def setUp(self) -> None:
        self.b = BaseParser()
        self.b.callbacks = {}

    def test_callbacks(self) -> None:
        called = 0

        def on_foo() -> None:
            nonlocal called
            called += 1

        self.b.set_callback("foo", on_foo)  # type: ignore[arg-type]
        self.b.callback("foo")  # type: ignore[arg-type]
        self.assertEqual(called, 1)

        self.b.set_callback("foo", None)  # type: ignore[arg-type]
        self.b.callback("foo")  # type: ignore[arg-type]
        self.assertEqual(called, 1)


class TestQuerystringParser(unittest.TestCase):
    def assert_fields(self, *args: tuple[bytes, bytes], **kwargs: Any) -> None:
        if kwargs.pop("finalize", True):
            self.p.finalize()

        self.assertEqual(self.f, list(args))
        if kwargs.get("reset", True):
            self.f: list[tuple[bytes, bytes]] = []

    def setUp(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.f = []

        name_buffer: list[bytes] = []
        data_buffer: list[bytes] = []

        def on_field_name(data: bytes, start: int, end: int) -> None:
            name_buffer.append(data[start:end])

        def on_field_data(data: bytes, start: int, end: int) -> None:
            data_buffer.append(data[start:end])

        def on_field_end() -> None:
            self.f.append((b"".join(name_buffer), b"".join(data_buffer)))

            del name_buffer[:]
            del data_buffer[:]

        self.p = QuerystringParser(
            callbacks={"on_field_name": on_field_name, "on_field_data": on_field_data, "on_field_end": on_field_end}
        )

    def test_simple_querystring(self) -> None:
        self.p.write(b"foo=bar")

        self.assert_fields((b"foo", b"bar"))

    def test_querystring_blank_beginning(self) -> None:
        self.p.write(b"&foo=bar")

        self.assert_fields((b"foo", b"bar"))

    def test_querystring_blank_end(self) -> None:
        self.p.write(b"foo=bar&")

        self.assert_fields((b"foo", b"bar"))

    def test_multiple_querystring(self) -> None:
        self.p.write(b"foo=bar&asdf=baz")

        self.assert_fields((b"foo", b"bar"), (b"asdf", b"baz"))

    def test_streaming_simple(self) -> None:
        self.p.write(b"foo=bar&")
        self.assert_fields((b"foo", b"bar"), finalize=False)

        self.p.write(b"asdf=baz")
        self.assert_fields((b"asdf", b"baz"))

    def test_streaming_break(self) -> None:
        self.p.write(b"foo=one")
        self.assert_fields(finalize=False)

        self.p.write(b"two")
        self.assert_fields(finalize=False)

        self.p.write(b"three")
        self.assert_fields(finalize=False)

        self.p.write(b"&asd")
        self.assert_fields((b"foo", b"onetwothree"), finalize=False)

        self.p.write(b"f=baz")
        self.assert_fields((b"asdf", b"baz"))

    def test_semicolon_is_data_not_a_field_separator(self) -> None:
        """Only `&` separates fields, as the WHATWG URL standard requires.

        Treating `;` as a separator lets a request smuggle a parameter past an
        upstream validator: a gateway sees `x=";role=admin"` while the parser
        used to see a second `role=admin` field, which wins in most frameworks.
        """
        self.p.write(b"role=user&x=;role=admin")

        self.assert_fields((b"role", b"user"), (b"x", b";role=admin"))

    def test_semicolon_is_data_regardless_of_chunking(self) -> None:
        """How the body is split across `write()` calls must not change the fields."""
        data = b"role=user&x=;role=admin"
        for first, last in split_all(data):
            self.reset()

            self.p.write(first)
            self.p.write(last)
            self.assert_fields((b"role", b"user"), (b"x", b";role=admin"))

    def test_semicolon_separated_body_parses_in_linear_time(self) -> None:
        """A semicolon-heavy body must not cost quadratic time.

        The parser used to scan the whole remaining buffer for `&` and only fall
        back to `;` once no `&` existed ahead, so a 1 MiB body of `a=1;` pairs
        cost O(B**2) byte comparisons - seconds of CPU for a single request.
        """
        data = b"a=1;" * 250_000

        start = time.monotonic()
        self.p.write(data)
        elapsed = time.monotonic() - start

        # Semicolons are data, so the body is one field whose value is the rest.
        self.assert_fields((b"a", data[2:]))
        assert elapsed < 5.0, "querystring parser rescanned the buffer per field (%.1fs)" % elapsed

    def test_semicolon_separated_names_parse_in_linear_time(self) -> None:
        """A body of bare `a;` names must not cost quadratic time either.

        Without an `=` in the chunk the parser stays in the field-name state,
        which used the same full-buffer `&` scan before falling back to `;`.
        """
        names = b"a;" * 500_000

        start = time.monotonic()
        self.p.write(names + b"=1")
        elapsed = time.monotonic() - start

        # Semicolons are data, so everything before the `=` is a single name.
        self.assert_fields((names, b"1"))
        assert elapsed < 5.0, "querystring parser rescanned the buffer per field name (%.1fs)" % elapsed

    def test_separator_search_is_bounded_by_max_size(self) -> None:
        """The separator search must stop at the truncation boundary.

        `write()` shortens the chunk to what `max_size` still allows, but the
        separator lookup used to scan the entire buffer, so a separator sitting
        past the budget dragged in data the size limit was meant to cut off.
        """
        self.p.max_size = 5

        # Only b"a=123" is within the budget; the ampersand at index 6 is past
        # it and must not be found.
        self.p.write(b"a=1234&b=2")

        self.assert_fields((b"a", b"123"))

    def test_equals_search_is_bounded_by_max_size(self) -> None:
        """The `=` search must also stop at the truncation boundary."""
        self.p.max_size = 3

        # Only b"abc" is within the budget; the equals sign at index 4 is past
        # it, so no field name/value split may be taken from beyond the limit.
        self.p.write(b"abcd=1")

        self.assert_fields()

    def test_too_large_field(self) -> None:
        self.p.max_size = 15

        # Note: len = 8
        self.p.write(b"foo=bar&")
        self.assert_fields((b"foo", b"bar"), finalize=False)

        # Note: len = 8, only 7 bytes processed
        self.p.write(b"a=123456")
        self.assert_fields((b"a", b"12345"))

    def test_invalid_max_size(self) -> None:
        with self.assertRaises(ValueError):
            p = QuerystringParser(max_size=-100)

    def test_strict_parsing_pass(self) -> None:
        data = b"foo=bar&another=asdf"
        for first, last in split_all(data):
            self.reset()
            self.p.strict_parsing = True

            print(f"{first!r} / {last!r}")

            self.p.write(first)
            self.p.write(last)
            self.assert_fields((b"foo", b"bar"), (b"another", b"asdf"))

    def test_strict_parsing_fail_double_sep(self) -> None:
        data = b"foo=bar&&another=asdf"
        for first, last in split_all(data):
            self.reset()
            self.p.strict_parsing = True

            cnt = 0
            with self.assertRaises(QuerystringParseError) as cm:
                cnt += self.p.write(first)
                cnt += self.p.write(last)
                self.p.finalize()

            # The offset should occur at 8 bytes into the data (as a whole),
            # so we calculate the offset into the chunk.
            if cm is not None:
                self.assertEqual(cm.exception.offset, 8 - cnt)

    def test_double_sep(self) -> None:
        data = b"foo=bar&&another=asdf"
        for first, last in split_all(data):
            print(f" {first!r} / {last!r} ")
            self.reset()

            cnt = 0
            cnt += self.p.write(first)
            cnt += self.p.write(last)

            self.assert_fields((b"foo", b"bar"), (b"another", b"asdf"))

    def test_strict_parsing_fail_no_value(self) -> None:
        self.p.strict_parsing = True
        with self.assertRaises(QuerystringParseError) as cm:
            self.p.write(b"foo=bar&blank&another=asdf")

        if cm is not None:
            self.assertEqual(cm.exception.offset, 8)

    def test_success_no_value(self) -> None:
        self.p.write(b"foo=bar&blank&another=asdf")
        self.assert_fields((b"foo", b"bar"), (b"blank", b""), (b"another", b"asdf"))

    def test_repr(self) -> None:
        # Issue #29; verify we don't assert on repr()
        _ignored = repr(self.p)


class TestOctetStreamParser(unittest.TestCase):
    def setUp(self) -> None:
        self.d: list[bytes] = []
        self.started = 0
        self.finished = 0

        def on_start() -> None:
            self.started += 1

        def on_data(data: bytes, start: int, end: int) -> None:
            self.d.append(data[start:end])

        def on_end() -> None:
            self.finished += 1

        self.p = OctetStreamParser(callbacks={"on_start": on_start, "on_data": on_data, "on_end": on_end})

    def assert_data(self, data: bytes, finalize: bool = True) -> None:
        self.assertEqual(b"".join(self.d), data)
        self.d = []

    def assert_started(self, val: bool = True) -> None:
        if val:
            self.assertEqual(self.started, 1)
        else:
            self.assertEqual(self.started, 0)

    def assert_finished(self, val: bool = True) -> None:
        if val:
            self.assertEqual(self.finished, 1)
        else:
            self.assertEqual(self.finished, 0)

    def test_simple(self) -> None:
        # Assert is not started
        self.assert_started(False)

        # Write something, it should then be started + have data
        self.p.write(b"foobar")
        self.assert_started()
        self.assert_data(b"foobar")

        # Finalize, and check
        self.assert_finished(False)
        self.p.finalize()
        self.assert_finished()

    def test_multiple_chunks(self) -> None:
        self.p.write(b"foo")
        self.p.write(b"bar")
        self.p.write(b"baz")
        self.p.finalize()

        self.assert_data(b"foobarbaz")
        self.assert_finished()

    def test_max_size(self) -> None:
        self.p.max_size = 5

        self.p.write(b"0123456789")
        self.p.finalize()

        self.assert_data(b"01234")
        self.assert_finished()

    def test_invalid_max_size(self) -> None:
        with self.assertRaises(ValueError):
            q = OctetStreamParser(max_size="foo")  # type: ignore[arg-type]


class TestBase64Decoder(unittest.TestCase):
    # Note: base64('foobar') == 'Zm9vYmFy'
    def setUp(self) -> None:
        self.f = BytesIO()
        self.d = Base64Decoder(self.f)

    def assert_data(self, data: bytes, finalize: bool = True) -> None:
        if finalize:
            self.d.finalize()

        self.f.seek(0)
        self.assertEqual(self.f.read(), data)
        self.f.seek(0)
        self.f.truncate()

    def test_simple(self) -> None:
        self.d.write(b"Zm9vYmFy")
        self.assert_data(b"foobar")

    def test_bad(self) -> None:
        with self.assertRaises(DecodeError):
            self.d.write(b"Zm9v!mFy")

    def test_split_properly(self) -> None:
        self.d.write(b"Zm9v")
        self.d.write(b"YmFy")
        self.assert_data(b"foobar")

    def test_bad_split(self) -> None:
        buff = b"Zm9v"
        for i in range(1, 4):
            first, second = buff[:i], buff[i:]

            self.setUp()
            self.d.write(first)
            self.d.write(second)
            self.assert_data(b"foo")

    def test_long_bad_split(self) -> None:
        buff = b"Zm9vYmFy"
        for i in range(5, 8):
            first, second = buff[:i], buff[i:]

            self.setUp()
            self.d.write(first)
            self.d.write(second)
            self.assert_data(b"foobar")

    def test_close_and_finalize(self) -> None:
        parser = Mock()
        f = Base64Decoder(parser)

        f.finalize()
        parser.finalize.assert_called_once_with()

        f.close()
        parser.close.assert_called_once_with()

    def test_bad_length(self) -> None:
        self.d.write(b"Zm9vYmF")  # missing ending 'y'

        with self.assertRaises(DecodeError):
            self.d.finalize()


class TestQuotedPrintableDecoder(unittest.TestCase):
    def setUp(self) -> None:
        self.f = BytesIO()
        self.d = QuotedPrintableDecoder(self.f)

    def assert_data(self, data: bytes, finalize: bool = True) -> None:
        if finalize:
            self.d.finalize()

        self.f.seek(0)
        self.assertEqual(self.f.read(), data)
        self.f.seek(0)
        self.f.truncate()

    def test_simple(self) -> None:
        self.d.write(b"foobar")
        self.assert_data(b"foobar")

    def test_with_escape(self) -> None:
        self.d.write(b"foo=3Dbar")
        self.assert_data(b"foo=bar")

    def test_with_newline_escape(self) -> None:
        self.d.write(b"foo=\r\nbar")
        self.assert_data(b"foobar")

    def test_with_only_newline_escape(self) -> None:
        self.d.write(b"foo=\nbar")
        self.assert_data(b"foobar")

    def test_with_split_escape(self) -> None:
        self.d.write(b"foo=3")
        self.d.write(b"Dbar")
        self.assert_data(b"foo=bar")

    def test_with_split_newline_escape_1(self) -> None:
        self.d.write(b"foo=\r")
        self.d.write(b"\nbar")
        self.assert_data(b"foobar")

    def test_with_split_newline_escape_2(self) -> None:
        self.d.write(b"foo=")
        self.d.write(b"\r\nbar")
        self.assert_data(b"foobar")

    def test_close_and_finalize(self) -> None:
        parser = Mock()
        f = QuotedPrintableDecoder(parser)

        f.finalize()
        parser.finalize.assert_called_once_with()

        f.close()
        parser.close.assert_called_once_with()

    def test_not_aligned(self) -> None:
        """
        https://github.com/andrew-d/python-multipart/issues/6
        """
        self.d.write(b"=3AX")
        self.assert_data(b":X")

        # Additional offset tests
        self.d.write(b"=3")
        self.d.write(b"AX")
        self.assert_data(b":X")

        self.d.write(b"q=3AX")
        self.assert_data(b"q:X")


# Load our list of HTTP test cases.
http_tests_dir = os.path.join(curr_dir, "test_data", "http")

# Read in all test cases and load them.
NON_PARAMETRIZED_TESTS = {"single_field_blocks"}
http_tests: list[TestParams] = []
for f in os.listdir(http_tests_dir):
    # Only load the HTTP test cases.
    fname, ext = os.path.splitext(f)
    if fname in NON_PARAMETRIZED_TESTS:
        continue

    if ext == ".http":
        # Get the YAML file and load it too.
        yaml_file = os.path.join(http_tests_dir, fname + ".yaml")

        # Load both.
        with open(os.path.join(http_tests_dir, f), "rb") as fh:
            test_data = fh.read()

        with open(yaml_file, "rb") as fy:
            yaml_data = yaml.safe_load(fy)

        http_tests.append({"name": fname, "test": test_data, "result": yaml_data})

# Datasets used for single-byte writing test.
single_byte_tests = [
    "almost_match_boundary",
    "almost_match_boundary_without_CR",
    "almost_match_boundary_without_LF",
    "almost_match_boundary_without_final_hyphen",
    "single_field_single_file",
]


EPILOGUE_TEST_HEAD = (
    "--boundary\r\n"
    'Content-Disposition: form-data; name="file"; filename="filename.txt"\r\n'
    "Content-Type: text/plain\r\n\r\n"
    "hello\r\n"
    "--boundary--"
).encode("latin-1")


def split_all(val: bytes) -> Iterator[tuple[bytes, bytes]]:
    """
    This function will split an array all possible ways.  For example:
        split_all([1,2,3,4])
    will give:
        ([1], [2,3,4]), ([1,2], [3,4]), ([1,2,3], [4])
    """
    for i in range(1, len(val) - 1):
        yield (val[:i], val[i:])


@parametrize_class
class TestFormParser(unittest.TestCase):
    def make(self, boundary: str | bytes, config: dict[str, Any] = {}) -> None:
        self.ended = False
        self.files: list[File] = []
        self.fields: list[Field] = []

        def on_field(f: FieldProtocol) -> None:
            self.fields.append(cast(Field, f))

        def on_file(f: FileProtocol) -> None:
            self.files.append(cast(File, f))

        def on_end() -> None:
            self.ended = True

        # Get a form-parser instance.
        self.f = FormParser("multipart/form-data", on_field, on_file, on_end, boundary=boundary, config=config)

    def assert_file_data(self, f: File, data: bytes) -> None:
        o = f.file_object
        o.seek(0)
        file_data = o.read()
        self.assertEqual(file_data, data)

    def assert_file(self, field_name: bytes, file_name: bytes, data: bytes) -> None:
        # Find this file.
        found = None
        for f in self.files:
            if f.field_name == field_name:
                found = f
                break

        # Assert that we found it.
        self.assertIsNotNone(found)
        assert found is not None

        try:
            # Assert about this file.
            self.assert_file_data(found, data)
            self.assertEqual(found.file_name, file_name)

            # Remove it from our list.
            self.files.remove(found)
        finally:
            # Close our file
            found.close()

    def assert_field(self, name: bytes, value: bytes) -> None:
        # Find this field in our fields list.
        found = None
        for f in self.fields:
            if f.field_name == name:
                found = f
                break

        # Assert that it exists and matches.
        self.assertIsNotNone(found)
        assert found is not None  # typing
        self.assertEqual(value, found.value)

        # Remove it for future iterations.
        self.fields.remove(found)

    @parametrize("param", http_tests)
    def test_http(self, param: TestParams) -> None:
        # Firstly, create our parser with the given boundary.
        boundary = param["result"]["boundary"]
        if isinstance(boundary, str):
            boundary = boundary.encode("latin-1")
        self.make(boundary)

        # Now, we feed the parser with data.
        exc = None
        try:
            processed = self.f.write(param["test"])
            self.f.finalize()
        except MultipartParseError as err:
            processed = 0
            exc = err

        # print(repr(param))
        # print("")
        # print(repr(self.fields))
        # print(repr(self.files))

        # Do we expect an error?
        if "error" in param["result"]["expected"]:
            self.assertIsNotNone(exc)
            assert exc is not None
            self.assertEqual(param["result"]["expected"]["error"], exc.offset)
            return

        # No error!
        self.assertEqual(processed, len(param["test"]), param["name"])

        # Assert that the parser gave us the appropriate fields/files.
        for e in param["result"]["expected"]:
            # Get our type and name.
            type = e["type"]
            name = e["name"].encode("latin-1")

            if type == "field":
                self.assert_field(name, e["data"])

            elif type == "file":
                self.assert_file(name, e["file_name"].encode("latin-1"), e["data"])

            else:
                assert False

    def test_random_splitting(self) -> None:
        """
        This test runs a simple multipart body with one field and one file
        through every possible split.
        """
        # Load test data.
        test_file = "single_field_single_file.http"
        with open(os.path.join(http_tests_dir, test_file), "rb") as f:
            test_data = f.read()

        # We split the file through all cases.
        for first, last in split_all(test_data):
            # Create form parser.
            self.make("boundary")

            # Feed with data in 2 chunks.
            i = 0
            i += self.f.write(first)
            i += self.f.write(last)
            self.f.finalize()

            # Assert we processed everything.
            self.assertEqual(i, len(test_data))

            # Assert that our file and field are here.
            self.assert_field(b"field", b"test1")
            self.assert_file(b"file", b"file.txt", b"test2")

    @parametrize("param", [t for t in http_tests if t["name"] in single_byte_tests])
    def test_feed_single_bytes(self, param: TestParams) -> None:
        """
        This test parses multipart bodies 1 byte at a time.
        """
        # Load test data.
        test_file = param["name"] + ".http"
        boundary = param["result"]["boundary"]
        with open(os.path.join(http_tests_dir, test_file), "rb") as f:
            test_data = f.read()

        # Create form parser.
        self.make(boundary)

        # Write all bytes.
        # NOTE: Can't simply do `for b in test_data`, since that gives
        # an integer when iterating over a bytes object on Python 3.
        i = 0
        for x in range(len(test_data)):
            b = test_data[x : x + 1]
            i += self.f.write(b)

        self.f.finalize()

        # Assert we processed everything.
        self.assertEqual(i, len(test_data))

        # Assert that the parser gave us the appropriate fields/files.
        for e in param["result"]["expected"]:
            # Get our type and name.
            type = e["type"]
            name = e["name"].encode("latin-1")

            if type == "field":
                self.assert_field(name, e["data"])

            elif type == "file":
                self.assert_file(name, e["file_name"].encode("latin-1"), e["data"])

            else:
                assert False

    def test_feed_blocks(self) -> None:
        """
        This test parses a simple multipart body 1 byte at a time.
        """
        # Load test data.
        test_file = "single_field_blocks.http"
        with open(os.path.join(http_tests_dir, test_file), "rb") as f:
            test_data = f.read()

        for c in range(1, len(test_data) + 1):
            # Skip first `d` bytes - not interesting
            for d in range(c):
                # Create form parser.
                self.make("boundary")
                # Skip
                i = 0
                self.f.write(test_data[:d])
                i += d
                for x in range(d, len(test_data), c):
                    # Write a chunk to achieve condition
                    #     `i == data_length - 1`
                    # in boundary search loop (multipatr.py:1302)
                    b = test_data[x : x + c]
                    i += self.f.write(b)

                self.f.finalize()

                # Assert we processed everything.
                self.assertEqual(i, len(test_data))

                # Assert that our field is here.
                self.assert_field(b"field", b"0123456789ABCDEFGHIJ0123456789ABCDEFGHIJ")

    def test_request_body_fuzz(self) -> None:
        """
        This test randomly fuzzes the request body to ensure that no strange
        exceptions are raised and we don't end up in a strange state.  The
        fuzzing consists of randomly doing one of the following:
            - Adding a random byte at a random offset
            - Randomly deleting a single byte
            - Randomly swapping two bytes
        """
        # Load test data.
        test_file = "single_field_single_file.http"
        with open(os.path.join(http_tests_dir, test_file), "rb") as f:
            test_data = f.read()

        iterations = 1000
        successes = 0
        failures = 0
        exceptions = 0

        print("Running %d iterations of fuzz testing:" % (iterations,))
        for i in range(iterations):
            # Create a bytearray to mutate.
            fuzz_data = bytearray(test_data)

            # Pick what we're supposed to do.
            choice = random.choice([1, 2, 3])
            if choice == 1:
                # Add a random byte.
                i = random.randrange(len(test_data))
                b = random.randrange(256)

                fuzz_data.insert(i, b)
                msg = "Inserting byte %r at offset %d" % (b, i)

            elif choice == 2:
                # Remove a random byte.
                i = random.randrange(len(test_data))
                del fuzz_data[i]

                msg = "Deleting byte at offset %d" % (i,)

            elif choice == 3:
                # Swap two bytes.
                i = random.randrange(len(test_data) - 1)
                fuzz_data[i], fuzz_data[i + 1] = fuzz_data[i + 1], fuzz_data[i]

                msg = "Swapping bytes %d and %d" % (i, i + 1)

            # Print message, so if this crashes, we can inspect the output.
            print("  " + msg)

            # Create form parser.
            self.make("boundary")

            # Feed with data, and ignore form parser exceptions.
            i = 0
            try:
                i = self.f.write(bytes(fuzz_data))
                self.f.finalize()
            except FormParserError:
                exceptions += 1
            else:
                if i == len(fuzz_data):
                    successes += 1
                else:
                    failures += 1

        print("--------------------------------------------------")
        print("Successes:  %d" % (successes,))
        print("Failures:   %d" % (failures,))
        print("Exceptions: %d" % (exceptions,))

    def test_request_body_fuzz_random_data(self) -> None:
        """
        This test will fuzz the multipart parser with some number of iterations
        of randomly-generated data.
        """
        iterations = 1000
        successes = 0
        failures = 0
        exceptions = 0

        print("Running %d iterations of fuzz testing:" % (iterations,))
        for i in range(iterations):
            data_size = random.randrange(100, 4096)
            data = os.urandom(data_size)
            print("  Testing with %d random bytes..." % (data_size,))

            # Create form parser.
            self.make("boundary")

            # Feed with data, and ignore form parser exceptions.
            i = 0
            try:
                i = self.f.write(bytes(data))
                self.f.finalize()
            except FormParserError:
                exceptions += 1
            else:
                if i == len(data):
                    successes += 1
                else:
                    failures += 1

        print("--------------------------------------------------")
        print("Successes:  %d" % (successes,))
        print("Failures:   %d" % (failures,))
        print("Exceptions: %d" % (exceptions,))

    def test_bad_start_boundary(self) -> None:
        self.make("boundary")
        data = b"--boundary\rfoobar"
        with self.assertRaises(MultipartParseError):
            self.f.write(data)

        self.make("boundary")
        data = b"--boundaryfoobar"
        with self.assertRaises(MultipartParseError):
            self.f.write(data)

        self.make("boundary")
        data = b"--Boundary\r\nfoobar"
        with self.assertRaisesRegex(MultipartParseError, "Expected boundary character %r, got %r" % (b"b"[0], b"B"[0])):
            self.f.write(data)

    def test_multipart_header_count_limit(self) -> None:
        self.make("poc")
        payload = b'--poc\r\nContent-Disposition: form-data; name="x"\r\n' + (b"X-A: 1\r\n" * 8)
        with self.assertRaisesRegex(MultipartParseError, "Maximum header count exceeded"):
            self.f.write(payload)

    def test_multipart_header_size_limit(self) -> None:
        self.make("poc")
        payload = b'--poc\r\nContent-Disposition: form-data; name="x"\r\n' + b"X-A: " + (b"a" * (4096 + 124))
        with self.assertRaisesRegex(MultipartParseError, "Maximum header size exceeded"):
            self.f.write(payload)

    def test_multipart_header_field_size_limit(self) -> None:
        # An oversized header name (no colon ever arrives) must also be bounded.
        self.make("poc")
        payload = b"--poc\r\n" + (b"X" * (4096 + 129))
        with self.assertRaisesRegex(MultipartParseError, "Maximum header size exceeded"):
            self.f.write(payload)

    def test_multipart_header_leading_whitespace_size_limit(self) -> None:
        # Whitespace skipped before a header value still counts towards the header size.
        self.make("poc")
        payload = b"--poc\r\nX-A:" + (b" " * (4096 + 128))
        with self.assertRaisesRegex(MultipartParseError, "Maximum header size exceeded"):
            self.f.write(payload)

    def test_multipart_header_limits_tracked_across_writes(self) -> None:
        # Splitting the headers over many write() calls must not reset the limits.
        self.make("poc")
        self.f.write(b'--poc\r\nContent-Disposition: form-data; name="x"\r\n')
        for _ in range(7):
            self.f.write(b"X-A: 1\r\n")
        with self.assertRaisesRegex(MultipartParseError, "Maximum header count exceeded"):
            self.f.write(b"X-A: 1\r\n")

        self.make("poc")
        self.f.write(b'--poc\r\nContent-Disposition: form-data; name="x"\r\nX-A: ')
        with self.assertRaisesRegex(MultipartParseError, "Maximum header size exceeded"):
            for _ in range(5000):
                self.f.write(b"a")

    def test_multipart_header_limits_config(self) -> None:
        payload = b'--poc\r\nContent-Disposition: form-data; name="x"\r\n' + (b"X-A: 1\r\n" * 8) + b"\r\nvalue\r\n--poc--"
        self.make("poc", config={"MAX_HEADER_COUNT": 9})
        self.f.write(payload)
        self.f.finalize()
        self.assert_field(b"x", b"value")

        self.make("poc", config={"MAX_HEADER_SIZE": 32})
        with self.assertRaisesRegex(MultipartParseError, "Maximum header size exceeded"):
            self.f.write(payload)

    def test_multipart_header_limits_reset_between_parts(self) -> None:
        # The per-part limits must reset on every part, while still being
        # tracked across the chunk boundaries of successive write() calls.
        self.make("boundary")

        long_value = b"a" * 4000
        data = b""
        for i in range(5):
            data += (
                b"--boundary\r\n"
                + (b'Content-Disposition: form-data; name="field%d"\r\n' % i)
                + b"Content-Type: text/plain\r\n"
                + b"X-Long: "
                + long_value
                + b"\r\n"
                + b"\r\n"
                + (b"value%d\r\n" % i)
            )
        data += b"--boundary--\r\n"

        # Feed in small chunks, so that a single header line spans several writes.
        for start in range(0, len(data), 512):
            self.f.write(data[start : start + 512])
        self.f.finalize()

        for i in range(5):
            self.assert_field(b"field%d" % i, b"value%d" % i)

    def test_octet_stream(self) -> None:
        files: list[File] = []

        def on_file(f: FileProtocol) -> None:
            files.append(cast(File, f))

        on_field = Mock()
        on_end = Mock()

        f = FormParser("application/octet-stream", on_field, on_file, on_end=on_end, file_name=b"foo.txt")
        self.assertTrue(isinstance(f.parser, OctetStreamParser))

        f.write(b"test")
        f.write(b"1234")
        f.finalize()

        # Assert that we only received a single file, with the right data, and that we're done.
        self.assertFalse(on_field.called)
        self.assertEqual(len(files), 1)
        self.assert_file_data(files[0], b"test1234")
        self.assertTrue(on_end.called)

    def test_querystring(self) -> None:
        fields: list[Field] = []

        def on_field(f: FieldProtocol) -> None:
            fields.append(cast(Field, f))

        on_file = Mock()
        on_end = Mock()

        def simple_test(f: FormParser) -> None:
            # Reset tracking.
            del fields[:]
            on_file.reset_mock()
            on_end.reset_mock()

            # Write test data.
            f.write(b"foo=bar")
            f.write(b"&test=asdf")
            f.finalize()

            # Assert we only received 2 fields...
            self.assertFalse(on_file.called)
            self.assertEqual(len(fields), 2)

            # ...assert that we have the correct data...
            self.assertEqual(fields[0].field_name, b"foo")
            self.assertEqual(fields[0].value, b"bar")

            self.assertEqual(fields[1].field_name, b"test")
            self.assertEqual(fields[1].value, b"asdf")

            # ... and assert that we've finished.
            self.assertTrue(on_end.called)

        f = FormParser("application/x-www-form-urlencoded", on_field, on_file, on_end=on_end)
        self.assertTrue(isinstance(f.parser, QuerystringParser))
        simple_test(f)

        f = FormParser("application/x-url-encoded", on_field, on_file, on_end=on_end)
        self.assertTrue(isinstance(f.parser, QuerystringParser))
        simple_test(f)

    def test_close_methods(self) -> None:
        parser = Mock()
        f = FormParser("application/x-url-encoded", None, None)
        f.parser = parser

        f.finalize()
        parser.finalize.assert_called_once_with()

        f.close()
        parser.close.assert_called_once_with()

    def test_bad_content_type(self) -> None:
        # We should raise a ValueError for a bad Content-Type
        with self.assertRaises(ValueError):
            f = FormParser("application/bad", None, None)

    def test_no_boundary_given(self) -> None:
        # We should raise a FormParserError when parsing a multipart message
        # without a boundary.
        with self.assertRaises(FormParserError):
            f = FormParser("multipart/form-data", None, None)

    def test_bad_content_transfer_encoding(self) -> None:
        data = (
            b'----boundary\r\nContent-Disposition: form-data; name="file"; filename="test.txt"\r\n'
            b"Content-Type: text/plain\r\n"
            b"Content-Transfer-Encoding: badstuff\r\n\r\n"
            b"Test\r\n----boundary--\r\n"
        )

        files: list[File] = []

        def on_file(f: FileProtocol) -> None:
            files.append(cast(File, f))

        on_field = Mock()
        on_end = Mock()

        # Test with erroring.
        config = {"UPLOAD_ERROR_ON_BAD_CTE": True}
        f = FormParser("multipart/form-data", on_field, on_file, on_end=on_end, boundary="--boundary", config=config)

        with self.assertRaises(FormParserError):
            f.write(data)
            f.finalize()

        # Test without erroring.
        config = {"UPLOAD_ERROR_ON_BAD_CTE": False}
        f = FormParser("multipart/form-data", on_field, on_file, on_end=on_end, boundary="--boundary", config=config)

        f.write(data)
        f.finalize()
        self.assert_file_data(files[0], b"Test")

    def test_handles_None_fields(self) -> None:
        fields: list[Field] = []

        def on_field(f: FieldProtocol) -> None:
            fields.append(cast(Field, f))

        on_file = Mock()
        on_end = Mock()

        f = FormParser("application/x-www-form-urlencoded", on_field, on_file, on_end=on_end)
        f.write(b"foo=bar&another&baz=asdf")
        f.finalize()

        self.assertEqual(fields[0].field_name, b"foo")
        self.assertEqual(fields[0].value, b"bar")

        self.assertEqual(fields[1].field_name, b"another")
        self.assertEqual(fields[1].value, None)

        self.assertEqual(fields[2].field_name, b"baz")
        self.assertEqual(fields[2].value, b"asdf")

    def test_urlencoded_semicolon_does_not_smuggle_field(self) -> None:
        fields: list[Field] = []

        def on_field(f: FieldProtocol) -> None:
            fields.append(cast(Field, f))

        f = FormParser("application/x-www-form-urlencoded", on_field, Mock())
        f.write(b"role=user&x=;role=admin")
        f.finalize()

        self.assertEqual([(x.field_name, x.value) for x in fields], [(b"role", b"user"), (b"x", b";role=admin")])

    def test_multipart_rfc_2231_params_do_not_override_plain_params(self) -> None:
        fields: list[Field] = []
        files: list[File] = []

        def on_field(f: FieldProtocol) -> None:
            fields.append(cast(Field, f))

        def on_file(f: FileProtocol) -> None:
            files.append(cast(File, f))

        f = FormParser("multipart/form-data", on_field, on_file, boundary="boundary")
        f.write(
            b"--boundary\r\n"
            b"Content-Disposition: form-data; name=\"role\"; name*=utf-8''admin\r\n\r\n"
            b"user\r\n"
            b"--boundary\r\n"
            b"Content-Disposition: form-data; name=\"file\"; filename=\"safe.txt\"; filename*=UTF-8''evil.php\r\n"
            b"Content-Type: text/plain\r\n\r\n"
            b"hello\r\n"
            b"--boundary--\r\n"
        )
        f.finalize()

        self.assertEqual([(x.field_name, x.value) for x in fields], [(b"role", b"user")])
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].field_name, b"file")
        self.assertEqual(files[0].file_name, b"safe.txt")

    @parametrize(
        "chunks",
        [
            [
                b"\r\nignored preamble\r\n"
                + (
                    b"--boundary\r\n"
                    b'Content-Disposition: form-data; name="file"; filename="filename.txt"\r\n'
                    b"Content-Type: text/plain\r\n\r\n"
                    b"hello\r\n"
                    b"--boundary--"
                )
            ],
            [
                b"\r\n" * 5_000_000
                + (
                    b"--boundary\r\n"
                    b'Content-Disposition: form-data; name="file"; filename="filename.txt"\r\n'
                    b"Content-Type: text/plain\r\n\r\n"
                    b"hello\r\n"
                    b"--boundary--"
                )
            ],
            [
                b"\r\n" * 5_000_000,
                (
                    b"--boundary\r\n"
                    b'Content-Disposition: form-data; name="file"; filename="filename.txt"\r\n'
                    b"Content-Type: text/plain\r\n\r\n"
                    b"hello\r\n"
                    b"--boundary--"
                ),
            ],
        ],
    )
    def test_multipart_parser_preamble_before_first_boundary(self, chunks: list[bytes]) -> None:
        """Parser must not hang or blow up on a preamble before the first boundary."""

        files: list[File] = []

        def on_file(f: FileProtocol) -> None:
            files.append(cast(File, f))

        f = FormParser("multipart/form-data", on_field=Mock(), on_file=on_file, boundary="boundary")
        for chunk in chunks:
            f.write(chunk)

        assert len(files) == 1
        self.assert_file_data(files[0], b"hello")

    def test_multipart_parser_data_after_last_boundary(self) -> None:
        """Parser must short-circuit on arbitrary epilogue data after the closing boundary (no O(N) scan)."""
        num = 50_000_000
        data = (
            "--boundary\r\n"
            'Content-Disposition: form-data; name="file"; filename="filename.txt"\r\n'
            "Content-Type: text/plain\r\n\r\n"
            "hello\r\n"
            "--boundary--" + "-" * num + "\r\n"
        )

        files: list[File] = []

        def on_file(f: FileProtocol) -> None:
            files.append(cast(File, f))

        f = FormParser("multipart/form-data", on_field=Mock(), on_file=on_file, boundary="boundary")
        f.write(data.encode("latin-1"))

    @parametrize(
        "chunks",
        [
            [EPILOGUE_TEST_HEAD + b"\r\n"],
            [EPILOGUE_TEST_HEAD + b"\r", b"\n"],
            [EPILOGUE_TEST_HEAD, b"\r\n"],
            [EPILOGUE_TEST_HEAD + b"\r\n--boundary\r\nthis is not a valid header\r\n\r\nnot a real part"],
        ],
    )
    def test_multipart_parser_ignores_epilogue(self, chunks: list[bytes]) -> None:
        """Epilogue data after the closing boundary must be ignored.

        Covers both the single-chunk case and the case where trailing CRLF is split across `write()` calls.
        The final case asserts that epilogue bytes are not parsed or validated.
        """
        files: list[File] = []

        def on_file(f: FileProtocol) -> None:
            files.append(cast(File, f))

        f = FormParser("multipart/form-data", on_field=Mock(), on_file=on_file, boundary="boundary")
        for chunk in chunks:
            f.write(chunk)

        assert len(files) == 1
        self.assert_file_data(files[0], b"hello")

    def _time_write(self, data: bytes) -> tuple[list[File], float]:
        """Feed `data` to a fresh multipart parser and return (files, elapsed_seconds)."""
        files: list[File] = []

        def on_file(f: FileProtocol) -> None:
            files.append(cast(File, f))

        f = FormParser("multipart/form-data", on_field=Mock(), on_file=on_file, boundary="boundary")
        start = time.monotonic()
        f.write(data)
        return files, time.monotonic() - start

    def test_multipart_parser_large_preamble_is_skipped_in_one_step(self) -> None:
        """A large CR/LF preamble must be skipped in one step, not walked byte by byte.

        The parser used to advance a single byte per leading CR/LF, so a body with a
        100 MB preamble burned tens of seconds of CPU before the first boundary.
        """
        data = b"\r\n" * 50_000_000 + EPILOGUE_TEST_HEAD

        files, elapsed = self._time_write(data)

        assert len(files) == 1
        self.assert_file_data(files[0], b"hello")
        assert elapsed < 5.0, "parser walked the preamble byte by byte (%.1fs)" % elapsed

    def test_multipart_parser_large_epilogue_is_skipped_in_one_step(self) -> None:
        """A large CRLF epilogue must be discarded in one step, not walked two bytes at a time.

        The parser used to consume the epilogue CRLF-pair by CRLF-pair, so a body with a
        50 MB trailing epilogue burned tens of seconds of CPU after the closing boundary.
        """
        data = EPILOGUE_TEST_HEAD + b"\r\n" * 25_000_000

        files, elapsed = self._time_write(data)

        assert len(files) == 1
        self.assert_file_data(files[0], b"hello")
        assert elapsed < 5.0, "parser walked the epilogue two bytes at a time (%.1fs)" % elapsed

    def test_max_size_multipart(self) -> None:
        # Load test data.
        test_file = "single_field_single_file.http"
        with open(os.path.join(http_tests_dir, test_file), "rb") as f:
            test_data = f.read()

        # Create form parser.
        self.make("boundary")

        # Set the maximum length that we can process to be halfway through the
        # given data.
        assert self.f.parser is not None
        self.f.parser.max_size = float(len(test_data)) / 2

        i = self.f.write(test_data)
        self.f.finalize()

        # Assert we processed the correct amount.
        self.assertEqual(i, len(test_data) / 2)

    def test_max_size_form_parser(self) -> None:
        # Load test data.
        test_file = "single_field_single_file.http"
        with open(os.path.join(http_tests_dir, test_file), "rb") as f:
            test_data = f.read()

        # Create form parser setting the maximum length that we can process to
        # be halfway through the given data.
        size = len(test_data) / 2
        self.make("boundary", config={"MAX_BODY_SIZE": size})

        i = self.f.write(test_data)
        self.f.finalize()

        # Assert we processed the correct amount.
        self.assertEqual(i, len(test_data) / 2)

    def test_octet_stream_max_size(self) -> None:
        files: list[File] = []

        def on_file(f: FileProtocol) -> None:
            files.append(cast(File, f))

        on_field = Mock()
        on_end = Mock()

        f = FormParser(
            "application/octet-stream",
            on_field,
            on_file,
            on_end=on_end,
            file_name=b"foo.txt",
            config={"MAX_BODY_SIZE": 10},
        )

        f.write(b"0123456789012345689")
        f.finalize()

        self.assert_file_data(files[0], b"0123456789")

    def test_invalid_max_size_multipart(self) -> None:
        with self.assertRaises(ValueError):
            MultipartParser(b"bound", max_size="foo")  # type: ignore[arg-type]

    def test_header_begin_callback(self) -> None:
        """
        This test verifies we call the `on_header_begin` callback.
        See GitHub issue #23
        """
        # Load test data.
        test_file = "single_field_single_file.http"
        with open(os.path.join(http_tests_dir, test_file), "rb") as f:
            test_data = f.read()

        calls = 0

        def on_header_begin() -> None:
            nonlocal calls
            calls += 1

        parser = MultipartParser("boundary", callbacks={"on_header_begin": on_header_begin}, max_size=1000)

        # Create multipart parser and feed it
        i = parser.write(test_data)
        parser.finalize()

        # Assert we processed everything.
        self.assertEqual(i, len(test_data))

        # Assert that we called our 'header_begin' callback three times; once
        # for each header in the multipart message.
        self.assertEqual(calls, 3)


class TestHelperFunctions(unittest.TestCase):
    def test_create_form_parser(self) -> None:
        r = create_form_parser({"Content-Type": b"application/octet-stream"}, None, None)
        self.assertTrue(isinstance(r, FormParser))

    def test_create_form_parser_error(self) -> None:
        headers: dict[str, bytes] = {}
        with self.assertRaises(ValueError):
            create_form_parser(headers, None, None)

    def test_parse_form(self) -> None:
        on_field = Mock()
        on_file = Mock()

        parse_form({"Content-Type": b"application/octet-stream"}, BytesIO(b"123456789012345"), on_field, on_file)

        assert on_file.call_count == 1

        # Assert that the first argument of the call (a File object) has size
        # 15 - i.e. all data is written.
        self.assertEqual(on_file.call_args[0][0].size, 15)

    def test_parse_form_content_length(self) -> None:
        files: list[FileProtocol] = []

        def on_field(field: FieldProtocol) -> None:
            pass

        def on_file(file: FileProtocol) -> None:
            files.append(file)

        parse_form(
            {"Content-Type": b"application/octet-stream", "Content-Length": b"10"},
            BytesIO(b"123456789012345"),
            on_field,
            on_file,
        )

        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].size, 10)  # type: ignore[attr-defined]


def suite() -> unittest.TestSuite:
    suite = unittest.TestSuite()
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestFile))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestParseOptionsHeader))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestBaseParser))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestQuerystringParser))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestOctetStreamParser))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestBase64Decoder))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestQuotedPrintableDecoder))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestFormParser))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestHelperFunctions))

    return suite
