"""Regression checks for event labels, single-event inputs, and optional HDF5."""

import builtins
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_data_tools", ROOT / "pymcpop-gw/data_tools.py")
data_tools = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data_tools)


class EventAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)

    def samples_fixture(self, names):
        prefix = str(self.base / "samples_")
        np.savetxt(prefix + "allNames.txt", names, fmt="%s")
        values = np.arange(len(names), dtype=float)[:, None] * 10 + np.arange(4) + 1
        for field, array in {
            "m1d_samples": values + 20,
            "m2d_samples": values,
            "dL_samples": values / 10,
            "dL_PE_prior": values / 100,
        }.items():
            np.save(prefix + field + ".npy", array)
        return prefix, values

    def gmm_fixture(self, names):
        prefix = str(self.base / "gmm_")
        count = len(names)
        np.savetxt(prefix + "allNames.txt", names, fmt="%s")
        np.savetxt(prefix + "allNgm.txt", np.ones(count, dtype=int), fmt="%d")
        markers = np.arange(count, dtype=float) + 10
        means = np.repeat(markers[:, None, None], 3, axis=2)
        covariance = np.tile(np.eye(3), (count, 1, 1, 1))
        for field, array in {
            "cho-means": means[:, 0],
            "cho-covs": covariance[:, 0],
            "gmm_log_wts": np.zeros((count, 1)),
            "gmm_means": means,
            "gmm_icovs": covariance,
            "gmm_covs": covariance,
            "gmm_cho_covs": covariance,
            "gmm_log_dets": np.zeros((count, 1)),
        }.items():
            np.save(prefix + field + ".npy", array)
        lower = np.repeat(markers[:, None], 3, axis=1)
        bounds = np.stack((lower, lower + 1), axis=-1)
        np.save(prefix + "gmm_fit_coord_bounds.npy", bounds)
        np.savetxt(prefix + "gmm_fit_coord_names.txt", ["logMc", "q", "logdL"], fmt="%s")
        Path(prefix + "gmm_fit_transform_info.json").write_text(
            json.dumps({"transform_mode": "event_bounded_flogit"})
        )
        return prefix, markers, bounds

    def whitelist(self, names):
        path = self.base / "selected.txt"
        np.savetxt(path, names, fmt="%s")
        return [str(path)]

    def quiet_load(self, function, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(divide="ignore"):
            return function(*args, **kwargs)

    def test_sample_whitelist_labels_follow_selected_rows_and_sample_cap(self):
        prefix, values = self.samples_fixture(["GW_A", "GW_B", "GW_C"])
        result = self.quiet_load(
            data_tools.load_data_samples, [prefix], nmax=2,
            events_use=self.whitelist(["GW_C", "GW_A"]),
        )
        self.assertEqual(result["allnames"], [["GW_A", "GW_C"]])
        np.testing.assert_array_equal(result["m2d_samples"], values[[0, 2], :2])
        np.testing.assert_array_equal(result["allNsamples"], [2, 2])

    def test_bounded_gmm_whitelist_aligns_names_means_and_bounds(self):
        prefix, markers, bounds = self.gmm_fixture(["GW_A", "GW_B", "GW_C"])
        result = self.quiet_load(
            data_tools.load_data_interp, [prefix], bounded_transform=True,
            events_use=self.whitelist(["GW_C", "GW_A"]),
        )
        self.assertEqual(result["allnames"], [["GW_A", "GW_C"]])
        np.testing.assert_array_equal(result["gmm_means"][:, 0, 0], markers[[0, 2]])
        np.testing.assert_array_equal(result["gmm_fit_coord_bounds"], bounds[[0, 2]])

    def test_single_event_text_files_load_in_both_formats(self):
        samples, values = self.samples_fixture(["GW_ONLY"])
        gmms, markers, bounds = self.gmm_fixture(["GW_ONLY"])
        sample_result = self.quiet_load(data_tools.load_data_samples, [samples])
        gmm_result = self.quiet_load(data_tools.load_data_interp, [gmms], bounded_transform=True)
        self.assertEqual(sample_result["allnames"], [["GW_ONLY"]])
        self.assertEqual(gmm_result["allnames"], [["GW_ONLY"]])
        np.testing.assert_array_equal(sample_result["m2d_samples"], values)
        np.testing.assert_array_equal(gmm_result["gmm_fit_coord_bounds"], bounds)

    def test_hdf5_dependency_is_optional_for_npy_and_explicit_for_hdf5(self):
        prefix, _ = self.samples_fixture(["GW_ONLY"])
        original_import = builtins.__import__

        def without_h5py(name, *args, **kwargs):
            if name == "h5py":
                raise ModuleNotFoundError("No module named h5py", name="h5py")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=without_h5py):
            result = self.quiet_load(data_tools.load_data_samples, [prefix])
            self.assertEqual(result["allnames"], [["GW_ONLY"]])
            with self.assertRaisesRegex(ImportError, "HDF5 posterior samples requires h5py"):
                self.quiet_load(data_tools.load_data_samples, ["missing.h5"])

    def test_injection_thinning_preserves_retained_spin_coordinates(self):
        formats = {
            "polar": {"chi1": "chi1", "chi2": "chi2", "ct1": "cost1", "ct2": "cost2"},
            "cartesian": {field: field for field in (
                "spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z",
            )},
        }
        for name, fields in formats.items():
            with self.subTest(format=name):
                prefix = str(self.base / (name + "_"))
                markers = np.arange(4, dtype=float)
                arrays = {
                    "m1d": markers + 10,
                    "m2d": markers + 1,
                    "dL": markers + 0.5,
                    "log_p_draw": -markers - 1,
                    "Ngen": np.asarray(100),
                }
                arrays.update({field: markers / 10 + 0.01 * i for i, field in enumerate(fields)})
                # Excluded coordinate products may coexist in historical data.
                # They must not interfere with retained polar/Cartesian loading.
                arrays.update({"chieff": markers / 10, "chip": markers / 10})
                for field, array in arrays.items():
                    np.save(prefix + field + ".npy", array)
                Path(prefix + "Tobs.txt").write_text("2.5\n")
                result = self.quiet_load(data_tools.load_injections, [prefix], allPercUse=[0.5])
                indices = (result["m1d"][0] - 10).astype(int)
                np.testing.assert_array_equal(result["Ndet"], [2])
                np.testing.assert_array_equal(result["Ngen"], [50])
                np.testing.assert_array_equal(result["Tobs"], [2.5])
                for source, returned in fields.items():
                    np.testing.assert_array_equal(result[returned][0], arrays[source][indices])
                np.testing.assert_array_equal(result["log_wt"][0], arrays["log_p_draw"][indices])
                self.assertNotIn("chieff", result)
                self.assertNotIn("chip", result)


if __name__ == "__main__":
    unittest.main()
