"""KV reuse requires known weight and service identities on both peers."""


def same_snapshot(source, target):
    return bool(isinstance(source, dict) and isinstance(target, dict)
                and isinstance(source.get("snapshot_id"), str) and source["snapshot_id"]
                and source["snapshot_id"] == target.get("snapshot_id")
                and type(source.get("optimizer_revision")) is int
                and type(target.get("optimizer_revision")) is int
                and source["optimizer_revision"] >= 0
                and source["optimizer_revision"] == target["optimizer_revision"])


def snapshot_identity(value):
    return {name: value.get(name) for name in (
        "snapshot_id", "optimizer_revision", "server_instance_id", "server_generation")}


def same_incarnation(source, current):
    """An old service generation cannot attest a cache after a reload/restart."""
    return bool(same_snapshot(source, current)
                and isinstance(source.get("server_instance_id"), str)
                and source["server_instance_id"]
                and source["server_instance_id"] == current.get("server_instance_id")
                and type(source.get("server_generation")) is int
                and type(current.get("server_generation")) is int
                and source["server_generation"] >= 0
                and source["server_generation"] == current["server_generation"])
