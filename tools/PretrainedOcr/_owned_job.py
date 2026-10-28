"""Private Windows containment for one supervised native process tree.

The root is assigned while its initial thread is suspended. Its new, non-
inheritable Job handle stays in the supervisor; descendants inherit membership,
including venv children and descendants whose parent exits. No PID-wide signal,
WMI query or existing controller Job is used for termination.
"""
import ctypes
from ctypes import wintypes
import subprocess
import time

CREATE_SUSPENDED = 0x4


class _BasicLimits(ctypes.Structure):
    _fields_ = [('process_time', ctypes.c_longlong), ('job_time', ctypes.c_longlong),
                ('flags', wintypes.DWORD), ('minimum_working_set', ctypes.c_size_t),
                ('maximum_working_set', ctypes.c_size_t), ('active_process_limit', wintypes.DWORD),
                ('affinity', ctypes.c_size_t), ('priority', wintypes.DWORD), ('scheduling', wintypes.DWORD)]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in
                ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [('basic', _BasicLimits), ('io', _IoCounters), ('process_memory', ctypes.c_size_t),
                ('job_memory', ctypes.c_size_t), ('peak_process_memory', ctypes.c_size_t), ('peak_job_memory', ctypes.c_size_t)]


class _Accounting(ctypes.Structure):
    _fields_ = [('user_time', ctypes.c_longlong), ('kernel_time', ctypes.c_longlong),
                ('period_user_time', ctypes.c_longlong), ('period_kernel_time', ctypes.c_longlong),
                ('page_faults', wintypes.DWORD), ('total_processes', wintypes.DWORD),
                ('active_processes', wintypes.DWORD), ('terminated_processes', wintypes.DWORD)]


class _ThreadEntry(ctypes.Structure):
    _fields_ = [('size', wintypes.DWORD), ('usage', wintypes.DWORD), ('thread_id', wintypes.DWORD),
                ('owner_pid', wintypes.DWORD), ('base_priority', wintypes.LONG),
                ('delta_priority', wintypes.LONG), ('flags', wintypes.DWORD)]


class _WindowsApi:
    def __init__(self):
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        signatures = {
            'CreateJobObjectW': ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            'SetInformationJobObject': ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
            'AssignProcessToJobObject': ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            'TerminateJobObject': ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            'QueryInformationJobObject': ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p], wintypes.BOOL),
            'CloseHandle': ([wintypes.HANDLE], wintypes.BOOL),
            'GetProcessId': ([wintypes.HANDLE], wintypes.DWORD),
            'CreateToolhelp32Snapshot': ([wintypes.DWORD, wintypes.DWORD], wintypes.HANDLE),
            'Thread32First': ([wintypes.HANDLE, ctypes.POINTER(_ThreadEntry)], wintypes.BOOL),
            'Thread32Next': ([wintypes.HANDLE, ctypes.POINTER(_ThreadEntry)], wintypes.BOOL),
            'OpenThread': ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            'GetProcessIdOfThread': ([wintypes.HANDLE], wintypes.DWORD),
            'ResumeThread': ([wintypes.HANDLE], wintypes.DWORD),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.kernel, name)
            function.argtypes, function.restype = args, result

    def checked(self, result):
        if not result:
            raise ctypes.WinError(ctypes.get_last_error())
        return result

    def create_job(self):
        handle = self.checked(self.kernel.CreateJobObjectW(None, None))
        limits = _ExtendedLimits()
        limits.basic.flags = 0x2000  # KILL_ON_JOB_CLOSE; no BREAKAWAY permission.
        try:
            self.checked(self.kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
        except BaseException:
            self.close(handle)
            raise
        return handle

    def close(self, handle):
        self.checked(self.kernel.CloseHandle(handle))

    def process_id(self, handle):
        return self.checked(self.kernel.GetProcessId(handle))

    def assign(self, job, process):
        self.checked(self.kernel.AssignProcessToJobObject(job, process))

    def threads(self, pid):
        snapshot = self.kernel.CreateToolhelp32Snapshot(0x4, 0)  # TH32CS_SNAPTHREAD
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        entry = _ThreadEntry()
        entry.size = ctypes.sizeof(entry)
        found = []
        try:
            present = self.kernel.Thread32First(snapshot, ctypes.byref(entry))
            while present:
                if entry.owner_pid == pid:
                    found.append(entry.thread_id)
                present = self.kernel.Thread32Next(snapshot, ctypes.byref(entry))
            if ctypes.get_last_error() != 18:  # ERROR_NO_MORE_FILES
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self.close(snapshot)
        return found

    def open_thread(self, tid):
        return self.checked(self.kernel.OpenThread(0x0002 | 0x0800, False, tid))

    def thread_owner(self, handle):
        return self.checked(self.kernel.GetProcessIdOfThread(handle))

    def resume(self, handle):
        count = self.kernel.ResumeThread(handle)
        if count == 0xffffffff:
            raise ctypes.WinError(ctypes.get_last_error())
        return count

    def terminate(self, job):
        self.checked(self.kernel.TerminateJobObject(job, 15))

    def active(self, job):
        value = _Accounting()
        self.checked(self.kernel.QueryInformationJobObject(job, 1, ctypes.byref(value), ctypes.sizeof(value), None))
        return value.active_processes


class NativeJob:
    """Keep one fresh kernel-owned tree attached to its original Popen handle."""
    def __init__(self, api=None):
        self.api = api or _WindowsApi()
        self.handle = self.api.create_job()
        self.assigned = False
        self.process_handle = None

    def close(self):
        if self.handle is not None:
            self.api.close(self.handle)
            self.handle = None

    def attach_and_resume(self, process):
        if self.process_handle is not None or process.poll() is not None:
            raise RuntimeError('Native containment requires one fresh suspended root')
        self.process_handle = process._handle  # Retained Popen OS handle, not a newly opened PID.
        if self.api.process_id(self.process_handle) != process.pid:
            raise RuntimeError('Native process handle identity differs')
        self.api.assign(self.handle, self.process_handle)
        self.assigned = True
        threads = self.api.threads(process.pid)
        if len(threads) != 1:
            raise RuntimeError('Suspended native root must have exactly one initial thread')
        thread = self.api.open_thread(threads[0])
        try:
            if (process.poll() is not None or self.api.process_id(self.process_handle) != process.pid
                    or self.api.thread_owner(thread) != process.pid):
                raise RuntimeError('Initial thread does not belong to the pinned native root')
            if self.api.resume(thread) != 1:
                raise RuntimeError('Initial native thread suspend count is ambiguous')
        finally:
            self.api.close(thread)

    def stop(self, process, deadline):
        """Confirm the whole job and root exit inside the caller's single bound."""
        if self.handle is None:
            raise RuntimeError('Confined native job handle is no longer available')
        def remaining():
            value = deadline-time.monotonic()
            if value <= 0:
                raise subprocess.TimeoutExpired('owned native job cleanup', 15)
            return value
        try:
            remaining()
            if self.assigned:
                if self.api.active(self.handle):
                    self.api.terminate(self.handle)
            else:
                if self.api.process_id(process._handle) != process.pid:
                    raise RuntimeError('Native process handle identity differs during cleanup')
                if process.poll() is None:
                    # Assignment failed before the sole initial thread was resumed.
                    process.kill()  # Original Popen handle, never a replacement PID.
            while self.api.active(self.handle):
                time.sleep(min(.01, remaining()))
            process.wait(timeout=remaining())
            remaining()
        finally:
            # On a failed query/wait, closing still contains our tree, but the
            # caller must retain the audit lease because closure was unproved.
            self.close()
