"""Settings regressions that do not import JAX or start an inference job."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pymcpop-gw"))
from settings_io import build_parser, parse_args_with_optional_settings


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = {"fin_data": ["data with spaces.h5"],
                     "fin_injections": ["injections"], "fin_priors": "priors.json",
                     "fout": "output", "nsteps": 10}

    def parse_json(self, values, extra=(), parser=None):
        path = Path(self.temp.name) / "input_args.json"
        path.write_text(json.dumps(values))
        return parse_args_with_optional_settings(parser or build_parser(),
                                                ["--settings", str(path), *extra])

    def test_typed_roundtrip_and_cli_precedence(self):
        values = dict(self.base, events_use=[], allTobs=None, nth=None,
                      fix_H0=False, s_local=0.5, settings="old/input_args.json")
        flags = self.parse_json(values, ["--nsteps", "2", "--fin_data", "new.h5"])
        self.assertEqual(flags.nsteps, 2)
        self.assertEqual(flags.fin_data, ["new.h5"])
        self.assertEqual(flags.events_use, [])
        self.assertIsNone(flags.allTobs)
        self.assertIsNone(flags.nth)
        self.assertEqual(flags.fix_H0, 0)
        self.assertEqual(flags.s_local, [0.5])
        self.assertEqual(flags.settings, str(Path(self.temp.name) / "input_args.json"))

    def test_ini_and_json_agree_with_quoted_paths(self):
        path = Path(self.temp.name) / "settings.ini"
        path.write_text('[settings]\nfin_data = "data with spaces.h5"\n'
                        'fin_injections = injections\nfin_priors = priors.json\n'
                        'fout = output\nnsteps = 10\n')
        ini = parse_args_with_optional_settings(build_parser(), ["--settings", str(path)])
        js = self.parse_json(self.base)
        self.assertEqual({k:v for k,v in vars(ini).items() if k != "settings"},
                         {k:v for k,v in vars(js).items() if k != "settings"})

    def test_unknown_removed_invalid_and_required_values_fail(self):
        cases = [dict(self.base, unknown_option=1),
                 dict(self.base, DP_conditional_z=0),
                 dict(self.base, mass_model="PL_DPUC"),
                 dict(self.base, rate_model="PL"),
                 dict(self.base, spin_model="default"),
                 dict(self.base, nsteps=1.5),
                 dict(self.base, fin_data=[]),
                 dict(self.base, fin_priors=None)]
        for values in cases:
            with self.subTest(values=values), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.parse_json(values)

    def test_parser_reuse_does_not_retain_file_defaults(self):
        parser = build_parser()
        self.assertEqual(self.parse_json(dict(self.base, nth=7), parser=parser).nth, 7)
        self.assertIsNone(self.parse_json(self.base, parser=parser).nth)
        self.assertTrue(next(a for a in parser._actions if a.dest == "fin_priors").required)

    def test_json_empty_optional_list_and_string_spaces_preserved(self):
        flags = self.parse_json(dict(self.base, events_exclude=[]))
        self.assertEqual(flags.fin_data, ["data with spaces.h5"])
        self.assertEqual(flags.events_exclude, [])

    def test_scalar_or_pair_controls_accept_both_json_representations(self):
        names = ("s_local", "q_mbound_mu_low", "q_mbound_mu_up",
                 "q_mbound_sig_low", "q_mbound_sig_up",
                 "L_small_res_percentile", "L_small_res_factor")
        for name in names:
            with self.subTest(name=name):
                scalar = self.parse_json(dict(self.base, **{name: 0.1}))
                pair = self.parse_json(dict(self.base, **{name: [0.1, 0.2]}))
                self.assertEqual(getattr(scalar, name), [0.1])
                self.assertEqual(getattr(pair, name), [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
