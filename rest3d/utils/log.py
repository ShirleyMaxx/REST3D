import datetime
import logging
import os
import sys

_ROOT_NAME = "rest3d"
_FMT = logging.Formatter("%(message)s")


def get_logger(stage_name=None):
    """Return a logger that writes ``%(message)s`` to stdout.

    Multiple calls with the same ``stage_name`` return the same logger and
    avoid duplicating the stdout handler.
    """
    name = _ROOT_NAME if stage_name is None else f"{_ROOT_NAME}.{stage_name}"
    logger = logging.getLogger(name)
    if not any(isinstance(h, logging.StreamHandler) and h.stream is sys.stdout
               for h in logger.handlers):
        logger.setLevel(logging.INFO)
        logger.propagate = False
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(_FMT)
        logger.addHandler(ch)
    return logger


def attach_phase_log(logger, log_dir, phase_name):
    """Attach a per-phase timestamped FileHandler. Returns handler for later detach."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"log_{phase_name}_{ts}.log")
    return attach_file_handler(logger, log_path)


def attach_file_handler(logger, log_path):
    """Attach a FileHandler that mirrors the logger output to ``log_path``.

    Returns the handler so the caller can detach it later via
    ``logger.removeHandler(h); h.close()``.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(_FMT)
    logger.addHandler(fh)
    return fh


def log_result(logger, child_names, best_iter_children, per_child_mean, per_child_best):
    def _f(v):
        return f"({v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f})"
    sources = [("best_iter", best_iter_children)]
    if per_child_mean is not None:
        sources.append(("final_mean", per_child_mean))
    if per_child_best is not None:
        sources.append(("final_best", per_child_best))
    W = 24
    hdr = (f"  {'':12s}  {'placed_origin':{W}s}  {'placed_world':{W}s}"
           f"  {'settled_origin':{W}s}  {'settled_world':{W}s}")
    sep = f"  {'-'*12}  {'-'*W}  {'-'*W}  {'-'*W}  {'-'*W}"
    for j, name in enumerate(child_names):
        logger.info(f"\n  [local_group table | {name}]")
        logger.info(hdr)
        logger.info(sep)
        for label, children in sources:
            d = children[j]
            logger.info(f"  {label:12s}  {_f(d['placed_origin']):{W}s}  "
                        f"{_f(d['placed_world']):{W}s}  "
                        f"{_f(d['settled_origin']):{W}s}  "
                        f"{_f(d['settled_world']):{W}s}")


def setup_run_logging(logger, output_dir: str, debug_log: bool = False) -> str:
    """Configure timestamped file + console handlers on *logger*.

    Also redirects stderr to the same log file so Isaac Gym output is captured.
    Returns the log file path.
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(output_dir, "log"), exist_ok=True)
    log_path = os.path.join(output_dir, "log", f"run_{ts}.log")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if debug_log else logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    _fd = os.open(log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    os.dup2(_fd, 2)
    os.close(_fd)
    return log_path
