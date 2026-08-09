#!/usr/bin/env python3
"""Read-only helper to identify SSH-reachable cluster nodes."""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import socket


def probe(host: str, *, port: int, timeout: float) -> str | None:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return host
    except OSError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hosts", nargs="+", help="hostnames or IP addresses to probe")
    parser.add_argument("--port", type=int, default=22, help="SSH port")
    parser.add_argument("--timeout", type=float, default=0.75)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    probe_host = functools.partial(probe, port=args.port, timeout=args.timeout)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        reachable = [host for host in executor.map(probe_host, args.hosts) if host]
    print("\n".join(reachable))


if __name__ == "__main__":
    main()
