"""Interactive input that can be cancelled without stranding an executor thread."""
from __future__ import annotations

import asyncio
import os
import sys
import threading


async def read_input(prompt: str) -> str:
    if os.name != "nt":
        # Canonical terminal input retains OS line editing. Read only available
        # bytes, without buffering a subsequent line or blocking the event loop.
        import select

        descriptor = sys.stdin.fileno()
        print(prompt, end="", flush=True)
        line = bytearray()
        while True:
            if not select.select([descriptor], [], [], 0)[0]:
                await asyncio.sleep(0.025)
                continue
            char = os.read(descriptor, 1)
            if not char:
                if not line:
                    raise EOFError
                break
            if char == b"\n":
                break
            line.extend(char)
        return line.decode(sys.stdin.encoding or "utf-8", errors="replace").rstrip("\r")

    # Retain builtin input's console line editing and stdin/EOF behavior. Unlike
    # to_thread(input), this reader is explicitly cancelled and joined before
    # asyncio.run shuts down. Never abandon a reader holding Python's stdin lock.
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenThread.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenThread.restype = wintypes.HANDLE
    kernel.CancelSynchronousIo.argtypes = (wintypes.HANDLE,)
    kernel.CancelSynchronousIo.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    result: list[str] = []
    errors: list[BaseException] = []

    def read() -> None:
        try:
            result.append(input(prompt))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=read, name="resident-stdin")
    thread.start()
    handle = kernel.OpenThread(0x0001, False, thread.native_id)  # THREAD_TERMINATE
    try:
        if not handle and thread.is_alive():
            raise ctypes.WinError(ctypes.get_last_error())
        while thread.is_alive():
            await asyncio.sleep(0.025)
        if errors:
            raise errors[0]
        return result[0]
    finally:
        async def finish_reader() -> None:
            try:
                # Cancellation can race with the read beginning. ERROR_NOT_FOUND
                # (1168) is normal: retry until completion, then join the reader.
                while thread.is_alive():
                    if not kernel.CancelSynchronousIo(handle):
                        code = ctypes.get_last_error()
                        if code != 1168:
                            raise ctypes.WinError(code)
                    await asyncio.sleep(0.025)
                thread.join()
            finally:
                if handle:
                    kernel.CloseHandle(handle)

        cleanup = asyncio.create_task(finish_reader())
        # A second Ctrl+C/task cancellation must not abandon the stdin thread.
        cancelled = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                cancelled = True
        cleanup.result()
        if cancelled:
            raise asyncio.CancelledError
