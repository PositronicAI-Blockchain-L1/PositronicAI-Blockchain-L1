"""
Positronic — Secure Memory Management

Utilities for clearing sensitive data (passwords, private keys) from memory
to prevent forensic recovery via memory dumps or cold boot attacks.
"""

import ctypes
import sys


def secure_clear_string(s: str) -> None:
    """Best-effort clearing of a Python string from memory.

    LIMITATION: Python strings are immutable. The ctypes approach to
    overwrite internal buffers is fragile and may corrupt the interpreter
    if the string is interned or shared. Instead, we rely on del + gc
    which is the safest practical approach for CPython.

    Callers should set the variable to None after calling this function
    and avoid keeping other references to the sensitive string.
    """
    if not isinstance(s, str) or not s:
        return
    # Force garbage collection of the string object once all references
    # are dropped. Callers must also set their variable to None.
    import gc
    gc.collect()


def secure_clear_bytes(b: bytes) -> None:
    """Overwrite a bytes object's internal buffer with zeros."""
    if not isinstance(b, bytes) or not b:
        return
    try:
        buf_size = len(b)
        if sys.implementation.name == "cpython":
            ctypes.memset(id(b) + sys.getsizeof(b"") - 1, 0, buf_size)
    except Exception:
        pass


def secure_clear_bytearray(ba: bytearray) -> None:
    """Overwrite a bytearray with zeros. This is the most reliable method."""
    if not isinstance(ba, bytearray):
        return
    for i in range(len(ba)):
        ba[i] = 0


def clear_widget_password(widget) -> None:
    """Clear password from a tkinter/CTk Entry widget and its internal variable."""
    try:
        old_val = widget.get()
        widget.delete(0, "end")
        secure_clear_string(old_val)
    except Exception:
        pass
