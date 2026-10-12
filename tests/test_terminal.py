from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class TerminalProcessTests(unittest.TestCase):
    """Exercise real blocking stdin and asyncio.run, not mocked executor calls."""

    def run_child(self, code: str, *, console: bool = False, stdin: str | None = None):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.txt"
            options = {}
            if console:
                startup = subprocess.STARTUPINFO()
                startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
                startup.wShowWindow = 0
                options.update(creationflags=subprocess.CREATE_NEW_CONSOLE,
                               startupinfo=startup)
            else:
                options.update(stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
            process = subprocess.Popen(
                [sys.executable, "-c", code, str(report)],
                cwd=Path(__file__).resolve().parents[1], **options)
            try:
                if console:
                    process.wait(timeout=8)
                    output = ""
                elif stdin is not None:
                    output, errors = process.communicate(stdin, timeout=8)
                    output += errors
                else:
                    process.wait(timeout=8)
                    output = process.stdout.read() + process.stderr.read()
                self.assertEqual(0, process.returncode, output)
                return report.read_text()
            finally:
                if process.poll() is None:
                    process.kill()  # Only our disposable regression fixture.
                    process.wait(timeout=5)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()

    def test_original_executor_input_waits_for_eof_after_task_cancellation(self):
        # Establish the failure mechanism: cancellation of an input future is
        # insufficient; asyncio.run still waits for its executor's blocked read.
        import time
        code = COMMON + """
async def main():
    task = asyncio.create_task(asyncio.to_thread(input, "owner> "))
    await asyncio.sleep(.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    report.write_text("task cancelled")
asyncio.run(main())
report.write_text("done")
"""
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.txt"
            process = subprocess.Popen([sys.executable, "-c", code, str(report)],
                                       cwd=Path(__file__).resolve().parents[1],
                                       stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
            try:
                deadline = time.monotonic() + 5
                while not report.exists() and time.monotonic() < deadline:
                    time.sleep(.025)
                self.assertEqual("task cancelled", report.read_text())
                time.sleep(.1)
                self.assertIsNone(process.poll())
                # Release only this fixture's pipe; no force or real application.
                process.stdin.close()
                process.wait(timeout=5)
                self.assertEqual("done", report.read_text())
            finally:
                process.stdin.close()
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)

    def test_idle_pipe_reader_cancellation_leaves_no_thread(self):
        self.assertEqual("done", self.run_child(CANCEL_READER))

    @unittest.skipUnless(os.name == "nt", "Windows console regression")
    def test_idle_console_reader_cancellation_leaves_no_thread(self):
        self.assertEqual("done", self.run_child(CANCEL_READER, console=True))

    @unittest.skipUnless(os.name == "nt", "Windows console regression")
    def test_ctrl_c_during_real_console_input_exits_asyncio_run(self):
        code = COMMON + '''
async def main():
    import ctypes
    ctypes.windll.kernel32.SetConsoleCtrlHandler(None, False)
    loop = asyncio.get_running_loop()
    loop.call_later(.2, lambda: ctypes.windll.kernel32.GenerateConsoleCtrlEvent(0, 0))
    await read_input("owner> ")
try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
assert not [t for t in threading.enumerate() if t is not threading.main_thread()]
report.write_text("done")
'''
        self.assertEqual("done", self.run_child(code, console=True))

    @unittest.skipUnless(os.name == "nt", "Windows console regression")
    def test_ctrl_c_cleans_up_actual_runtime_host_with_idle_stdin(self):
        code = COMMON + HOST + """
async def main():
    import ctypes
    ctypes.windll.kernel32.SetConsoleCtrlHandler(None, False)
    asyncio.get_running_loop().call_later(
        .2, lambda: ctypes.windll.kernel32.GenerateConsoleCtrlEvent(0, 0))
    try:
        await host.run()
    finally:
        assert host._stopping
        assert runtime.checkpoint is None
try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
assert not [t for t in threading.enumerate() if t is not threading.main_thread()]
report.write_text("done")
"""
        self.assertEqual("done", self.run_child(code, console=True))

    def test_repeated_cancellation_cannot_abandon_reader_cleanup(self):
        code = COMMON + """
async def main():
    task = asyncio.create_task(read_input("owner> "))
    await asyncio.sleep(.1)
    task.cancel()
    for _ in range(10):
        await asyncio.sleep(0)
        task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert not [t for t in threading.enumerate() if t is not threading.main_thread()]
asyncio.run(main())
report.write_text("done")
"""
        self.assertEqual("done", self.run_child(code))

    def test_lines_quit_and_eof_preserve_stdin(self):
        code = COMMON + '''
async def main():
    lines = [await read_input("owner> "), await read_input("owner> ")]
    try:
        await read_input("owner> ")
    except EOFError:
        lines.append("EOF")
    return lines
report.write_text(repr(asyncio.run(main())))
'''
        self.assertEqual("['hello', '/quit', 'EOF']",
                         self.run_child(code, stdin="hello\n/quit\n"))


COMMON = '''
import asyncio, sys, threading
from pathlib import Path
from resident.terminal import read_input
report = Path(sys.argv[1])
'''

CANCEL_READER = COMMON + '''
async def main():
    for delay in (0, 0, 0, .1, .1, .1):
        task = asyncio.create_task(read_input("owner> "))
        await asyncio.sleep(delay)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert not [t for t in threading.enumerate() if t is not threading.main_thread()]
    assert asyncio.get_running_loop()._default_executor is None
asyncio.run(main())
report.write_text("done")
'''


HOST = """
from types import SimpleNamespace
from resident.host import RuntimeHost, InstancePolicy
class IdleRuntime:
    owner = SimpleNamespace(address_name="owner")
    checkpoint = None
    async def enqueue_startup_wakeups(self, queue):
        pass
    async def scheduler_loop(self, queue, stop):
        await stop.wait()
    async def output_dispatcher_loop(self, stop):
        await stop.wait()
    def observe_event_loop_lag(self, *args):
        pass
    def bind_event_loop_lag_checkpoint(self, callback):
        self.checkpoint = callback
runtime = IdleRuntime()
host = RuntimeHost({"resident": runtime},
                   {"resident": InstancePolicy(frozenset())},
                   SimpleNamespace(pending=lambda: ()))
"""
