import atexit
import contextlib
import io
import json
import os
import re
import sys
import traceback

from simsopt.field import BiotSavart
from simsopt.geo import BoozerSurface, Surface

from .tags import generate_json_filename

class DriverLog:
    def __init__(self, log_path=None, *, comment_prefix="# ", enabled=True):
        self.enabled = enabled
        self.comment_prefix = comment_prefix
        self._fh = None
        self._stdout = sys.__stdout__ or sys.stdout
        if log_path and enabled:
            self._fh = open(log_path, "w")
            atexit.register(self.close)

    def __call__(self, line="", data=False):
        if not self.enabled:
            return
        text = ("" if data else self.comment_prefix) + str(line)
        print(text, file=self._stdout, flush=True)
        if self._fh is not None:
            self._fh.write(text + "\n")
            self._fh.flush()

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def write_raw(self, text):
        if self._fh is not None:
            self._fh.write(text)
            self._fh.flush()

class ErrStream:
    def __init__(self, log, stderr):
        self._log = log
        self._stderr = stderr
    
    def write(self, s):
        self._stderr.write(s)
        self._log.write_raw(s)
        return len(s)

    def flush(self):
        self._stderr.flush()

    def isatty(self):
        return False
    
    def fileno(self):
        return self._stderr.fileno()

class LogStream:
    def __init__(self, log, *, tap=None):
        self._log = log
        self._buffer = ""
        self._tap = tap

    def write(self, s):
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._log(line)
            if self._tap is not None:
                self._tap(line)
        return len(s)
    
    def flush(self):
        if self._buffer:
            self._log(self._buffer)
            self._buffer = ""

    def isatty(self):
        return False
    
    def fileno(self):
        raise io.UnsupportedOperation("fileno")

@contextlib.contextmanager
def stdout_to_log(log, *, tap=None, capture_stderr=True):
    out_stream = LogStream(log, tap=tap)
    err_redirect = (contextlib.redirect_stderr(ErrStream(log, sys.stderr))
                    if capture_stderr else contextlib.nullcontext())
    try:
        with contextlib.redirect_stdout(out_stream), err_redirect:
            try:
                yield
            finally:
                out_stream.flush()
    except BaseException:
        log.write_raw(traceback.format_exc())
        raise

def save_to_json(
    obj: BoozerSurface | BiotSavart | Surface,
    tag_dict: dict
):
    savefile = generate_json_filename(tag_dict)
    if isinstance(obj, BoozerSurface):
        err = False
        for key in ["biotsavart", "surface", "boozersurface"]:
            if key not in tag_dict:
                print(f"Missing tags for {key}")
                err = True
        assert (not err), "Missing required tags"
        obj.save(savefile)
        if not obj.need_to_run_code:
            res = obj.res
            iota = res["iota"]
            G = res["G"]
            statefile = savefile.replace("boozersurface", "state")
            with open(statefile, "w") as f:
                json.dump(dict(iota=iota, G=G), f, indent=2)
    elif isinstance(obj, (BiotSavart, Surface)):
        err = False
        if (
            (isinstance(obj, BiotSavart) and ("biotsavart" not in tag_dict)) or
            (isinstance(obj, Surface) and ("surface" not in tag_dict))
        ):
            print(f"Missing tags for {type(obj).__name__}")
            err = True
        assert (not err), "Missing required tags"
        obj.save(savefile)
    else:
        raise ValueError(f"Unsupported object type: {type(obj)}")
    
    return savefile

    
