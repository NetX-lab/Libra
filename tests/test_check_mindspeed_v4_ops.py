import ctypes
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from engine.mindspeed_v4_ops import (
    OPTIONAL_SPARSE_MLA_SYMBOLS,
    REQUIRED_SPARSE_MLA_SYMBOLS,
    discover_cann_libraries,
    inspect_sparse_mla_symbols,
)


class InspectLibrariesTest(unittest.TestCase):
    def test_inspect_libraries_finds_required_symbols(self):
        class FakeLibrary:
            pass

        library = FakeLibrary()
        for symbol in REQUIRED_SPARSE_MLA_SYMBOLS:
            setattr(library, symbol, object())
        with patch.object(ctypes, "CDLL", return_value=library):
            found, errors = inspect_sparse_mla_symbols([Path("libopapi_transformer.so")])

        self.assertEqual(found, set(REQUIRED_SPARSE_MLA_SYMBOLS))
        self.assertEqual(errors, [])

    def test_inspect_libraries_reports_missing_symbols(self):
        class FakeLibrary:
            aclnnSparseFlashMla = object()

        with patch.object(ctypes, "CDLL", return_value=FakeLibrary()):
            found, errors = inspect_sparse_mla_symbols([Path("libopapi_transformer.so")])

        self.assertEqual(found, {"aclnnSparseFlashMla"})
        self.assertEqual(errors, [])

    def test_910b3_required_symbols_do_not_include_ascend950_grad_metadata(self):
        self.assertTrue(set(OPTIONAL_SPARSE_MLA_SYMBOLS).isdisjoint(REQUIRED_SPARSE_MLA_SYMBOLS))
        class FakeLibrary:
            pass

        library = FakeLibrary()
        for symbol in REQUIRED_SPARSE_MLA_SYMBOLS:
            setattr(library, symbol, object())
        with patch.object(ctypes, "CDLL", return_value=library):
            found, errors = inspect_sparse_mla_symbols([Path("libopapi_transformer.so")])
        missing = set(REQUIRED_SPARSE_MLA_SYMBOLS) - found
        self.assertEqual(missing, set())
        self.assertEqual(errors, [])

    def test_discover_direct_cann_ops_root(self):
        with tempfile.TemporaryDirectory() as temp:
            library = Path(temp) / "aarch64-linux" / "lib64" / "libopapi_transformer.so"
            library.parent.mkdir(parents=True)
            library.touch()
            self.assertEqual(discover_cann_libraries(Path(temp)), [library])


if __name__ == "__main__":
    unittest.main()
