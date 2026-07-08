import atexit
import contextlib
import io
import json
import numpy as np
import os
import sys
import traceback
import warnings

from simsopt.field import BiotSavart
from simsopt.geo import BoozerSurface, Surface

from .tags import (
    generate_biotsavart_filename,
    generate_surface_filename,
    generate_boozersurface_filename,
)

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
    tag_dict: dict,
    minimal: bool = True,
    enforce_tags: bool = True,
    out_dir: str = ".",
    overwrite: bool = False,
):
    statefile = None
    statefile_exists = False
    if isinstance(obj, BoozerSurface):
        err = False
        for key in ["biotsavart", "surface", "boozersurface"]:
            if key not in tag_dict:
                print(f"Missing tags for {key}")
                err = True
        assert (not err), "Missing required tags"
        savefile = generate_boozersurface_filename(tag_dict, minimal=minimal, enforce_tags=enforce_tags)
        if not obj.need_to_run_code:
            state_tag_dict = dict(
                biotsavart=tag_dict["biotsavart"],
                surface=tag_dict["surface"],
                boozersurface=dict(
                    tag="state",
                    constraint_weight_str=tag_dict["boozersurface"]["constraint_weight_str"],
                    volume_target_str=tag_dict["boozersurface"]["volume_target_str"],
                ),
            )
            for key, val in tag_dict.items():
                if key not in state_tag_dict:
                    state_tag_dict[key] = val
            statefile = generate_boozersurface_filename(state_tag_dict, minimal=minimal, enforce_tags=enforce_tags)
            statefile = os.path.join(out_dir, statefile)
            res = obj.res
            iota = res["iota"]
            G = res["G"]
            targetlabel = obj.targetlabel
            statefile_exists = os.path.exists(statefile)
    elif isinstance(obj, (BiotSavart, Surface)):
        err = False
        if (
            (isinstance(obj, BiotSavart) and ("biotsavart" not in tag_dict)) or
            (isinstance(obj, Surface) and ("surface" not in tag_dict))
        ):
            print(f"Missing tags for {type(obj).__name__}")
            err = True
        assert (not err), "Missing required tags"
        if isinstance(obj, BiotSavart):
            savefile = generate_biotsavart_filename(tag_dict, minimal=minimal, enforce_tags=enforce_tags)
        else:
            savefile = generate_surface_filename(tag_dict, minimal=minimal, enforce_tags=enforce_tags)
    else:
        raise ValueError(f"Unsupported object type: {type(obj)}")
    savefile = os.path.join(out_dir, savefile)
    if os.path.exists(savefile):
        if overwrite:
            warnings.warn(f"Save file {savefile} already exists and will be overwritten.")
        else:
            raise FileExistsError(f"Save file {savefile} already exists. Use `overwrite=True` to overwrite existing files.")
    obj.save(savefile)
    if statefile:
        if statefile_exists: # Fine to overwrite, just notify user but no need to warn
            print(f"State file {statefile} already exists and will be overwritten.")
        with open(statefile, "w") as f:
            json.dump(dict(iota=iota, G=G, targetlabel=targetlabel), f, indent=2)
    
    return savefile

def read_poincare_npz(file):
    poincare = np.load(file, allow_pickle=True)

    res_phi_hits_flat = poincare["res_phi_hits_flat"]
    res_phi_hits_to_plot_dict = {}
    for iline, *row in res_phi_hits_flat:
        iline = int(iline)
        if iline not in res_phi_hits_to_plot_dict:
            res_phi_hits_to_plot_dict[iline] = [row]
        else:
            res_phi_hits_to_plot_dict[iline].append(row)
    res_phi_hits = []
    for iline, rows in res_phi_hits_to_plot_dict.items():
        res_phi_hits.append(np.asarray(rows))

    return res_phi_hits
