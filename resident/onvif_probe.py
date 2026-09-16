from __future__ import annotations

import argparse
import asyncio

from .config import _cameras_from_environment
from .onvif import OnvifClient


async def _probe(camera_id: str, pulls: int, request_timeout: float, pull_timeout: float) -> int:
    camera = next((item for item in _cameras_from_environment() if item.id == camera_id), None)
    if camera is None or camera.onvif is None:
        print(f"Camera {camera_id!r} does not have explicit ONVIF configuration.")
        return 2
    client = OnvifClient(
        camera.onvif, request_timeout=request_timeout, pull_timeout=pull_timeout)
    pullpoint = None
    try:
        service, topics, schemas = await client.discover()
        print(f"Advertised event topics ({len(topics)}):")
        for topic in topics:
            print(f"- {topic}")
        print(f"Advertised property schemas ({len(schemas)}):")
        for schema in schemas:
            print(f"- {schema.topic} (IsProperty=true)")
            for section_name, fields in (
                    ("Source", schema.source), ("Key", schema.key), ("Data", schema.data)):
                if fields:
                    print(f"  {section_name}: " + ", ".join(
                        f"{field.name}:{field.type}" for field in fields))
        pullpoint = await client.subscribe(service)
        print("PullPoint subscription active.")
        for _ in range(pulls):
            for notification in await client.pull(pullpoint):
                print(
                    f"Observed event shape: topic={notification['topic']!r}, "
                    f"fields={notification['fields']!r}")
    except Exception as exc:
        print(f"ONVIF probe failed ({type(exc).__name__}); endpoint and credentials were not shown.")
        return 1
    finally:
        if pullpoint is not None:
            try:
                await client.unsubscribe(pullpoint)
            except Exception:
                print("The best-effort ONVIF unsubscribe failed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe one explicitly configured camera's ONVIF event service")
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--pulls", type=int, default=3)
    parser.add_argument("--request-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--pull-timeout-seconds", type=float, default=5.0)
    args = parser.parse_args()
    return asyncio.run(_probe(
        args.camera_id, max(1, min(args.pulls, 20)),
        max(0.1, args.request_timeout_seconds), max(1.0, args.pull_timeout_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
