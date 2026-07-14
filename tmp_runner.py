import subprocess
import sys
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


print_lock = threading.Lock()


def list_dvc_targets(prefix: str = "run-assay@") -> list[str]:
    result = subprocess.run(
        ["uv", "run", "dvc", "stage", "list", "--name-only"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    targets = sorted([
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().startswith(prefix)
    ])

    if not targets:
        raise RuntimeError(f"No DVC targets found with prefix {prefix!r}")

    return targets


def run_target(target: str) -> None:
    proc = subprocess.Popen(
        [
            "uv", "run", "dvc",
            "repro",
            "--force",
            "--single-item",
            target,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None

    for line in proc.stdout:
        with print_lock:
            sys.stdout.write(f"[{target}] {line}")
            sys.stdout.flush()

    return_code = proc.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, proc.args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--prefix", type=str, default="run-assay@")
    args = parser.parse_args()

    targets = list_dvc_targets(prefix=args.prefix)

    print(f"Found {len(targets)} targets matching {args.prefix!r}")

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {
            ex.submit(run_target, target): target
            for target in targets
        }

        for fut in as_completed(futures):
            target = futures[fut]

            try:
                fut.result()
            except Exception as exc:
                with print_lock:
                    print(f"[{target}] failed: {exc}", file=sys.stderr)
                raise
            else:
                with print_lock:
                    print(f"[{target}] done")


if __name__ == "__main__":
    main()