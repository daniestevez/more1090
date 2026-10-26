#!/usr/bin/env python3
"""Compare the benchmark results in the working tree with those in git.

This checks that a new run of run_benchmark.py has reproduced the results
that are committed to git, up to the fields that depend on machine load and
wall-clock time (runtime and throughput), which are expected to vary.
"""

import json
import pathlib
import subprocess
import sys

# Fields that vary between runs due to machine load and wall-clock time.
IGNORED_FIELDS = {
    "real_time_duration",
    "throughput_sps",
    "thread_throughput_sps",
}


def remove_ignored(obj):
    """Recursively remove the ignored fields from a parsed JSON object."""
    if isinstance(obj, dict):
        return {k: remove_ignored(v) for k, v in obj.items() if k not in IGNORED_FIELDS}
    if isinstance(obj, list):
        return [remove_ignored(v) for v in obj]
    return obj


def differences(path, a, b):
    """Return the paths at which a and b differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        ret = []
        for k in a.keys() | b.keys():
            composite_path = f"{path}.{k}" if path else k
            if k not in a:
                ret.append(f"{composite_path}: missing in git version")
            elif k not in b:
                ret.append(f"{composite_path}: missing in working tree")
            else:
                ret += differences(f"{composite_path}", a[k], b[k])
        return ret
    if a != b:
        return [f"{path}: working tree {a!r} != git {b!r}"]
    return []


def main():
    results_dir = pathlib.Path(__file__).parent / "results"
    toplevel = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    worktree_files = {f.name for f in results_dir.glob("*.json")}
    git_files = {
        pathlib.Path(p).name
        for p in subprocess.run(
            ["git", "ls-files", str(results_dir)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    }

    errors = 0
    for name in sorted(worktree_files | git_files):
        f = results_dir / name
        if name not in git_files:
            print(f"{name}: not committed to git")
            errors += 1
            continue
        if name not in worktree_files:
            print(f"{name}: missing from working tree")
            errors += 1
            continue
        git_contents = subprocess.run(
            ["git", "show", f"HEAD:{f.relative_to(toplevel).as_posix()}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        diffs = differences(
            None,
            remove_ignored(json.loads(f.read_text())),
            remove_ignored(json.loads(git_contents)),
        )
        if diffs:
            print(f"{name}: DIFFERS")
            for d in diffs:
                print(f"  {d}")
            errors += 1

    if errors:
        sys.exit(f"{errors} file(s) differ from git HEAD")
    print("all results match git HEAD (up to runtime/throughput)")


if __name__ == "__main__":
    main()
