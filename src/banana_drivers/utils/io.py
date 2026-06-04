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

def resolve_filename(filename):
    # Filename cases:
    # [1] {tag}.banana_dofs.yaml
    # [2] {tag}.{boozersurface|state|biotsavart|surface}.json
    # [3] {tag}.{boozersurface|state|biotsavart|surface}.v{version_number}.json
    # [4] {tag}.{boozersurface|state|biotsavart|surface}.v{version_number}.iter{iter_number}.json
    # {tag} links to a table or database with metadata about the file:
    # 1. Type of object: BiotSavart, Surface, BoozerSurface (linked to a BiotSavart tag and a Surface tag, object is paired with a 'state' file)
    # 2. BiotSavart:
    #   2.1  Source ('original'|'angled'|'angled_flipped'|<generated_tag>)
    #   2.2  TF coil current fixed status
    #   2.3  TF coil current if fixed, else None
    #   2.4  Banana coil current fixed status
    #   2.5  Banana coil current if fixed, else None
    #   2.6  Proxy coil current (always fixed)
    #   2.7  VF coil current fixed status
    #   2.8  VF coil current if fixed, else None
    #   2.9  Banana coil order
    #   2.10 Banana coil number of quadpoints
    # 3. Surface:
    #   3.1 Source ('original'|'flipped'|<generated_tag>)
    #   3.2 mpol
    #   3.3 ntor
    #   3.4 Number of phi quadpoints
    #   3.5 Number of theta quadpoints
    # 4. BoozerSurface:
    #   4.1 Optimization label {standard|virtual_casing|finite_build}
    #       The standard optimization path is:
    #         i. Assemble BoozerSurface from initial BiotSavart and Surface objects, rescaling currents and resizing resolution according to inputs. If a tag already exists with the exact same parameters, raise Error.
    #        ii. Run initial Boozer solve.
    #       iii. Run stage 2.
    #        iv. Run singlestage.
    #         v. If continuation, assemble new BoozerSurface with new tag and run singlestage. Loop continuation as needed.
    #       The virtual casing path only accepts pre-assembled BoozerSurface objects under the assumption that they have already gone through some optimization.
    #         i. Run fixed-boundary VMEC with the standard HBT tokamak plasma current profile so that `curtor = boozersurface.biotsavart.coils[PROXY_IDX].current.get_value()`.
    #        ii. Run virtual casing.
    #       iii. Rescale the proxy coil current to 0.
    #        iv. Run stage 2 with the virtual casing squared-flux target.
    #       For now, we're only using the finite-build method for post-optimization analysis.
    #         i. Assemble finite-build BiotSavart extracted from an optimized BoozerSurface object.
    #        ii. Reassemble the BoozerSurface.
    #       iii. Pass to analysis scripts.
    #   4.2 BiotSavart source tag
    #   4.3 Surface source tag
    #   4.4 Constraint weight (BoozerExact if 0 or None else BoozerLS)
    #   4.5 Initial iota guess 
    #   4.6 Initial G sign (No G value, |G| is accessed via BiotSavart TF coils)
    #   4.7 Volume target
    # BoozerSurface is paired with a state file that contains:
    # 1. Iota after solve
    # 2. Sign G after solve
    # 3. Volume after solve
    # BiotSavart and Surface objects will only act as inputs to BoozerSurface.
    # All other actions in banana_drivers will be performed using BoozerSurface objects.
    # The status of a particular input seed will be listed in a table e.g. initial Boozer solve (pre-stage 2), stage 2 optimized (pre-singlestage), singlestage.
    # If a BoozerSurface object's parameters are changed such that it triggers a new tag to be generated, the BiotSavart and Surface objects will be saved as inputs so that the new BoozerSurface object's tag will refer to the original tag.
    # For example:
    # new_tag, new_bsurf = rebuild_boozersurface(old_tag.old_bsurf)
    # new_tag --> old_tag.BiotSavart, old_tag.Surface

    pattern_case1 = r"^(?P<tag>[a-zA-Z0-9]+)\.banana_dofs\.yaml$"
    pattern_case2 = r"^(?P<tag>[a-zA-Z0-9]+)\.(?P<ftype>boozersurface|state|biotsavart|surface)\.json$"
    pattern_case3 = r"^(?P<tag>[a-zA-Z0-9]+)\.(?P<ftype>boozersurface|state|biotsavart|surface)\.v(?P<version_number>\d+)\.json$"
    pattern_case4 = r"^(?P<tag>[a-zA-Z0-9]+)\.(?P<ftype>boozersurface|state|biotsavart|surface)\.v(?P<version_number>\d+)\.iter(?P<iter_number>\d+)\.json$"
    patterns = [pattern_case1, pattern_case2, pattern_case3, pattern_case4]
    any_match = False
    for case, pattern in enumerate(patterns, start=1):
        match = re.match(pattern, filename)
        if match:
            any_match = True
            return case, match.groupdict()
    if not any_match:
        raise ValueError(f"Filename does not match any known pattern: {filename}")

# def resolve_filename(filename):
#     # Filename cases:
#     # [1] {prefix}.banana_dofs.yaml
#     # [2] {prefix1}.{prefix2}.{boozersurface|state}.init.json
#     # [3] {prefix1}.{prefix2}.{boozersurface|state}.{stage2_opt|opt}.v{version}.json
#     # [4] {prefix1}.{prefix2}.{boozersurface|state}.{stage2_opt|opt}.v{version}.iter{iter_number}.json
#     # [5] {prefix}.{biotsavart|surface}.init.json
#     # [6] {prefix}.{biotsavart|surface}.opt.v{version}.json
#     # [7] {prefix}.{biotsavart|surface}.opt.v{version}.iter{iter_number}.json
#     base = os.path.basename(filename)
#     base_split = base.split('.')
#     if (len(base_split) < 3) or (len(base_split) > 7):
#         raise ValueError(
# f"""Invalid filename: {filename}
# Filename convention:
#     <prefix>.banana_dofs.yaml
#     <prefix1>.<prefix2>.<boozersurface|state>.init.json
#     <prefix1>.<prefix2>.<boozersurface|state>.<stage2_opt|opt>.v<version>.json
#     <prefix1>.<prefix2>.<boozersurface|state>.<stage2_opt|opt>.v<version>.iter<iter_number>.json
#     <prefix>.<biotsavart|surface>.init.json
#     <prefix>.<biotsavart|surface>.opt.v<version>.json
#     <prefix>.<biotsavart|surface>.opt.v<version>.iter<iter_number>.json
# """)
    
#     case = 0
#     if len(base_split) == 3:
#         case = 1
#         prefix, ftype, ext = base_split
#         err_msg = f"Invalid filename: {filename}, expected format: <prefix>.banana_dofs.yaml"
#         assert (ftype == "banana_dofs") and (ext == "yaml"), err_msg
#         return case, prefix
#     elif len(base_split) == 4:
#         case = 5
#         prefix, ftype, init, ext = base_split
#         err_msg = f"Invalid filename: {filename}, expected format: <prefix>.<biotsavart|surface>.init.json"
#         assert (ftype in ["biotsavart", "surface"]) and (init == "init") and (ext == "json"), err_msg
#         return case, prefix
#     elif len(base_split) == 5:
#         if base_split[1] in ["biotsavart", "surface"]:
#             case = 6
#             prefix, ftype, opt, version, ext = base_split
#             version_match = re.match(r"v(\d+)", version)
#             err_msg = f"Invalid filename: {filename}, expected format: <prefix>.<biotsavart|surface>.opt.v<version>.json"
#             assert (opt == "opt") and version_match and (ext == "json"), err_msg
#             version_number = int(version_match.group(1))
#             return case, prefix, version_number
#         elif base_split[2] in ["boozersurface", "state"]:
#             case = 2
#             prefix1, prefix2, ftype, init, ext = base_split
#             err_msg = f"Invalid filename: {filename}, expected format: <prefix1>.<prefix2>.<boozersurface|state>.init.json"
#             assert (init in ["init", "init_solve"]) and (ext == "json"), err_msg
#             return case, prefix1, prefix2
#         else:
#             raise ValueError(
# f"""Invalid filename: {filename}
# Parsing [{base_split}]
# Either
#     {base_split[2]} not in ['biotsavart', 'surface'] or
#     {base_split[3]} not in ['boozersurface', 'state']
# For split length = 5 filename, excepted:
#     <prefix1>.<prefix2>.<boozersurface|state>.init.json
#     <prefix>.<biotsavart|surface>.opt.v<version>.json
# """)
#     elif len(base_split) == 6:
#         if base_split[1] in ["biotsavart", "surface"]:
#             case = 7
#             prefix, ftype, opt, version, iter_str, ext = base_split
#             version_match = re.match(r"v(\d+)", version)
#             iter_match = re.match(r"iter(\d+)", iter_str)
#             err_msg = f"Invalid filename: {filename}, expected format: <prefix>.<biotsavart|surface>.opt.v<version>.iter<iter_number>.json"
#             assert (opt == "opt") and version_match and iter_match and (ext == "json"), err_msg
#             version_number = int(version_match.group(1))
#             iter_number = int(iter_match.group(1))
#             return case, prefix, version_number, iter_number
#         elif base_split[2] in ["boozersurface", "state"]:
#             case = 3
#             prefix1, prefix2, ftype, opt, version, ext = base_split
#             version_match = re.match(r"v(\d+)", version)
#             err_msg = f"Invalid filename: {filename}, expected format: <prefix1>.<prefix2>.<boozersurface|state>.opt.v<version>.json"
#             assert (opt in ["init", "init_solve", "stage2_opt", "opt"]) and version_match and (ext == "json"), err_msg
#             version_number = int(version_match.group(1))
#             return case, prefix1, prefix2, version_number
#         else:
#             raise ValueError(
# f"""Invalid filename: {filename}
# For split length = 6 filename, excepted:
#     <prefix>.<biotsavart|surface>.opt.v<version>.iter<iter_number>.json
#     <prefix1>.<prefix2>.<boozersurface|state>.opt.v<version>.iter<iter_number>.json
# """)
#     elif len(base_split) == 7:
#         case = 4
#         prefix1, prefix2, ftype, opt, version, iter_str, ext = base_split
#         err_msg = f"Invalid filename: {filename}, expected format: <prefix1>.<prefix2>.<boozersurface|state>.opt.v<version>.iter<iter_number>.json"
#         assert (ftype in ["boozersurface", "state"]) and (opt in ["init", "init_solve", "stage2_opt", "opt"]) and re.match(r"v(\d+)", version) and re.match(r"iter(\d+)", iter_str) and (ext == "json"), err_msg
#         version_number = int(re.match(r"v(\d+)", version).group(1))
#         iter_number = int(re.match(r"iter(\d+)", iter_str).group(1))
#         return case, prefix1, prefix2, version_number, iter_number

def generate_savefile(
    ftype: str,
    prefix: str,
    prefix2: str | None = None,
    init_opt: str = "opt",
    version_number: int | None = None,
    iter_number: int | None = None,
    out_dir: str = "",
    ext: str = ".json"
) -> str:
    if ftype in ["boozersurface", "state"]:
        if prefix2 is None:
            raise ValueError("prefix2 must be provided for BoozerSurface objects.")
    savefile = os.path.join(out_dir, f"{prefix}")
    if prefix2 is not None:
        savefile += f".{prefix2}"
    savefile += f".{ftype}.{init_opt}"
    if version_number is not None:
        savefile += f".v{version_number}"
    if iter_number is not None:
        savefile += f".iter{iter_number}"
    savefile += ext
    return savefile

def save_to_json(
    obj: BoozerSurface | BiotSavart | Surface,
    prefix: str,
    prefix2: str | None = None,
    init_opt: str = "opt",
    version_number: int | None = None,
    iter_number: int | None = None,
    out_dir: str = "",
    ext: str = ".json"
):
    kwargs = dict(
        prefix2=prefix2,
        init_opt=init_opt,
        version_number=version_number,
        iter_number=iter_number,
        out_dir=out_dir,
        ext=ext,
    )
    if isinstance(obj, BoozerSurface):
        ftype = "boozersurface"
        savefile = generate_savefile(ftype, prefix, **kwargs)
        obj.save(savefile)
        if not obj.need_to_run_code:
            res = obj.res
            iota = res["iota"]
            G = res["G"]
            savefile = savefile.replace("boozersurface", "state")
            with open(savefile, "w") as f:
                json.dump(dict(iota=iota, G=G), f, indent=2)
    elif isinstance(obj, (BiotSavart, Surface)):
        ftype = "biotsavart" if isinstance(obj, BiotSavart) else "surface"
        savefile = generate_savefile(ftype, prefix, **kwargs)
        obj.save(savefile)
    else:
        raise ValueError(f"Unsupported object type: {type(obj)}")
    
    return savefile

    
