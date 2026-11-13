"""Reject overlapping lifecycle RPCs before any model/provider side effects."""
from functools import wraps
from threading import Lock


class RPCBusyError(RuntimeError):
    """Another RPC owns this instance; this request made no state changes."""


def lifecycle_lock():
    return Lock()


def exclusive_rpc(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        lock = self._rpc_lock
        if lock is None:  # Explicit legacy replay keeps its original behavior.
            return method(self, *args, **kwargs)
        if not lock.acquire(blocking=False):
            raise RPCBusyError('Another lifecycle RPC is in progress; request rejected without mutation')
        try:
            return method(self, *args, **kwargs)
        finally:
            lock.release()
    return call
