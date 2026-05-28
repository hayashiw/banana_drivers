import atexit

class DriverLog:
    """Callable logger: stdout always, optional tee to a log file.

    Non-data lines get a comment prefix in both sinks, so the log file doubles
    as a '#'-commented data file. Construct once per driver:

        log = DriverLog(os.path.join(out_dir, "log.txt"))
        log("INITIAL STATE")        # -> "# INITIAL STATE"
        log(csv_row, data=True)     # -> bare row
    """

    def __init__(self, log_path=None, *, comment_prefix="# ", enabled=True):
        self.enabled = enabled
        self.comment_prefix = comment_prefix
        self._fh = None
        if log_path and enabled:
            self._fh = open(log_path, "w")
            atexit.register(self.close)

    def __call__(self, line="", data=False):
        if not self.enabled:
            return
        text = ("" if data else self.comment_prefix) + str(line)
        print(text, flush=True)
        if self._fh is not None:
            self._fh.write(text + "\n")
            self._fh.flush()

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None