#!/usr/bin/env python3
"""Keep Linux page cache small for named files on Spark UMA hosts.

On GB10 unified memory, clean file-cache pages of freshly read model files
count against torch.cuda.mem_get_info free, which breaches the resident
stage loader's 12 GiB reserve. The loader reclaims via posix_fadvise
DONTNEED only at its own checkpoints; this gardener drops the same clean
pages continuously between those checkpoints. Read-only advisory: no
writes, no deletes, no root, no process interference.
"""
import json
import os
import sys
import time


def main():
    interval = float(sys.argv[1])
    paths = [Path(p) for p in sys.argv[2:]]
    while True:
        dropped_bytes = 0
        errors = 0
        for path in paths:
            try:
                fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                try:
                    dropped_bytes += os.fstat(fd).st_size
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
            except OSError:
                errors += 1
        print(json.dumps({"event": "gardener_pass", "files": len(paths),
                          "advised_bytes": dropped_bytes, "errors": errors}), flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    main()
