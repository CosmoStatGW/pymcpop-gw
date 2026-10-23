"""Check catalogue-reader imports without importing the fitting stack."""

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
source = ast.parse((ROOT / "pymcpop-gw/fit_posterior_samples.py").read_text())
reader_function = next(
    node for node in source.body
    if isinstance(node, ast.FunctionDef) and node.name == "_load_catalog_reader"
)
# Execute this dependency-loading helper alone: no NumPyro, GMM fitting, plots,
# catalogue files, or optional fitting dependencies are needed for these checks.
namespace = {"importlib": importlib}
exec(compile(ast.Module(body=[reader_function], type_ignores=[]), "reader_helper", "exec"), namespace)
load_reader = namespace["_load_catalog_reader"]


class ReaderImportTests(unittest.TestCase):
    def test_installed_package_layout(self):
        expected = object()
        with mock.patch.object(importlib, "import_module", return_value=SimpleNamespace(O4aData=expected)):
            self.assertIs(load_reader("O4adata", "O4aData"), expected)

    def test_explicit_pythonpath_legacy_layout(self):
        expected = object()
        missing_package = ModuleNotFoundError("missing package", name="MGCosmoPop")
        with mock.patch.object(
            importlib, "import_module",
            side_effect=[missing_package, SimpleNamespace(O4aData=expected)],
        ):
            self.assertIs(load_reader("O4adata", "O4aData"), expected)

    def test_missing_reader_reports_external_requirement(self):
        with mock.patch.object(
            importlib, "import_module",
            side_effect=[
                ModuleNotFoundError("missing package", name="MGCosmoPop"),
                ModuleNotFoundError("missing package", name="dataStructures"),
            ],
        ):
            with self.assertRaisesRegex(ImportError, "MGCosmoPop reader O4adata.O4aData.*PYTHONPATH"):
                load_reader("O4adata", "O4aData")

    def test_reader_dependency_failure_is_not_hidden_by_fallback(self):
        missing_dependency = ModuleNotFoundError("missing dependency", name="astropy")
        with mock.patch.object(importlib, "import_module", side_effect=missing_dependency):
            with self.assertRaises(ModuleNotFoundError) as result:
                load_reader("O4adata", "O4aData")
        self.assertIs(result.exception, missing_dependency)


if __name__ == "__main__":
    unittest.main()
