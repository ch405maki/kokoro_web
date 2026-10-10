"""Tests for the multi-worker launcher. No server, no torch, no sockets.

    .venv\\Scripts\\python.exe -m unittest tests.test_launcher -v
"""

from __future__ import annotations

import io
import os
import sys
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from app import launcher


class TestWorkerCount(unittest.TestCase):
    def test_defaults_to_one(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(launcher.resolve_workers(), 1)

    def test_environment_variable_is_used(self):
        with mock.patch.dict(os.environ, {launcher.WORKERS_ENV: "3"}, clear=True):
            self.assertEqual(launcher.resolve_workers(), 3)

    def test_command_line_beats_the_environment(self):
        with mock.patch.dict(os.environ, {launcher.WORKERS_ENV: "3"}, clear=True):
            self.assertEqual(launcher.resolve_workers("2"), 2)

    def test_blank_command_line_falls_back_to_the_environment(self):
        with mock.patch.dict(os.environ, {launcher.WORKERS_ENV: "2"}, clear=True):
            self.assertEqual(launcher.resolve_workers(""), 2)

    def test_surrounding_space_is_tolerated(self):
        with mock.patch.dict(os.environ, {launcher.WORKERS_ENV: " 4 "}, clear=True):
            self.assertEqual(launcher.resolve_workers(), 4)

    def test_is_capped_at_the_cpu_count(self):
        # More processes than cores cannot help, and 32 interpreters each holding
        # a copy of the model would exhaust memory instead of failing loudly.
        with mock.patch.dict(os.environ, {launcher.WORKERS_ENV: "32"}, clear=True):
            self.assertEqual(launcher.resolve_workers(cpus=8), 8)

    def test_a_hand_report_of_zero_cpus_still_gives_one_worker(self):
        # os.cpu_count() can return None on a platform it cannot determine.
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(launcher.resolve_workers("4", cpus=0), 1)

    def test_nonsense_is_rejected(self):
        with mock.patch.dict(os.environ, {launcher.WORKERS_ENV: "two"}, clear=True):
            with self.assertRaises(ValueError):
                launcher.resolve_workers()

    def test_zero_and_negative_are_rejected(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            for bad in ("0", "-2"):
                with self.assertRaises(ValueError):
                    launcher.resolve_workers(bad)


class TestThreadBudget(unittest.TestCase):
    """N workers must not each ask torch for every core."""

    def test_one_worker_is_left_alone(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(launcher.apply_thread_budget(1, cpus=16))
            self.assertNotIn(launcher.THREADS_ENV, os.environ)

    def test_the_cpus_are_split_evenly(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(launcher.apply_thread_budget(4, cpus=16), 4)
            self.assertEqual(os.environ[launcher.THREADS_ENV], "4")

    def test_an_odd_split_leaves_a_remainder_rather_than_zero(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(launcher.apply_thread_budget(3, cpus=4), 1)

    def test_more_workers_than_cpus_still_gets_one_thread_each(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(launcher.apply_thread_budget(8, cpus=2), 1)

    def test_an_explicit_thread_count_wins(self):
        with mock.patch.dict(os.environ, {launcher.THREADS_ENV: "6"}, clear=True):
            self.assertIsNone(launcher.apply_thread_budget(4, cpus=16))
            self.assertEqual(os.environ[launcher.THREADS_ENV], "6")

    def test_applying_it_twice_changes_nothing(self):
        # main() calls it for the banner and serve() calls it again; the second
        # call must not halve the budget a second time.
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(launcher.apply_thread_budget(2, cpus=16), 8)
            self.assertIsNone(launcher.apply_thread_budget(2, cpus=16))
            self.assertEqual(os.environ[launcher.THREADS_ENV], "8")


class _FakeUvicorn(types.ModuleType):
    """Stands in for the real module so no socket is ever opened."""

    def __init__(self) -> None:
        super().__init__("uvicorn")
        self.calls: list[tuple[object, dict]] = []
        self.threads_at_start: str | None = None

    def run(self, app, **kwargs) -> None:  # noqa: ANN001 - mirrors uvicorn.run
        self.calls.append((app, kwargs))
        self.threads_at_start = os.environ.get(launcher.THREADS_ENV)


class TestServe(unittest.TestCase):
    def test_the_app_is_passed_as_an_import_string(self):
        # uvicorn refuses an app object when workers > 1 - handing it one exits
        # with "You must pass the application as an import string to enable
        # 'reload' or 'workers'", so this is what makes the setting real.
        fake = _FakeUvicorn()
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.dict(sys.modules, {"uvicorn": fake}):
            launcher.serve("127.0.0.1", 8000, 3)
        app, kwargs = fake.calls[0]
        self.assertEqual(app, "app.server:app")
        self.assertEqual(kwargs["workers"], 3)
        self.assertEqual(kwargs["port"], 8000)

    def test_the_thread_budget_is_in_place_before_the_server_starts(self):
        # app.engine reads KOKORO_THREADS at import time and the workers inherit
        # this process's environment, so it has to be exported first.
        fake = _FakeUvicorn()
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.dict(sys.modules, {"uvicorn": fake}), \
                mock.patch.object(launcher, "cpu_count", return_value=16):
            launcher.serve("127.0.0.1", 8000, 4)
        self.assertEqual(fake.threads_at_start, "4")

    def test_one_worker_keeps_the_whole_cpu(self):
        fake = _FakeUvicorn()
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.dict(sys.modules, {"uvicorn": fake}), \
                mock.patch.object(launcher, "cpu_count", return_value=16):
            launcher.serve("127.0.0.1", 8000, 1)
        self.assertIsNone(fake.threads_at_start)


class TestCommandLine(unittest.TestCase):
    def test_a_bad_worker_count_is_reported_without_starting_anything(self):
        err = io.StringIO()
        with redirect_stderr(err), mock.patch.object(launcher, "serve") as serve:
            code = launcher.main(["--workers", "lots"])
        self.assertEqual(code, 2)
        self.assertIn("whole number", err.getvalue())
        serve.assert_not_called()

    def test_the_resolved_worker_count_is_passed_on(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                redirect_stdout(io.StringIO()), \
                mock.patch.object(launcher, "serve") as serve:
            self.assertEqual(launcher.main(["--port", "9100", "--workers", "2"]), 0)
        serve.assert_called_once()
        host, port, workers = serve.call_args.args
        self.assertEqual((host, port, workers), ("127.0.0.1", 9100, 2))

    def test_the_banner_names_the_workers_when_there_is_more_than_one(self):
        out = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=True), \
                redirect_stdout(out), mock.patch.object(launcher, "serve"):
            launcher.main(["--port", "9100", "--workers", "2"])
        self.assertIn("Workers: 2 processes", out.getvalue())

    def test_a_single_worker_is_not_mentioned(self):
        out = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=True), \
                redirect_stdout(out), mock.patch.object(launcher, "serve"):
            launcher.main(["--port", "9100"])
        self.assertNotIn("Workers:", out.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
