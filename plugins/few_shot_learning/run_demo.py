"""
Demo script for Few-Shot Learning plugin.

This script loads a dataset and launches the FiftyOne App.
Open the 'Few-Shot Learning' panel to start mining samples.

Supported models:
    - RocchioPrototypeModel: Centroid-based prototype classifier

Usage:
    python run_demo.py
"""

import os
import fiftyone as fo
import fiftyone.zoo as foz


# NOTE: This repository contains a plugin at `plugins/io/` whose Python module
# name is `io`. When plugin contexts are built, that module can be imported as
# `io` and shadow the Python stdlib `io` module. Some downstream dependencies
# (notably `boto3/botocore`, imported by FiftyOne zoo utilities) expect the
# stdlib module and will crash if it's been shadowed.
#
# We keep this demo script resilient by restoring stdlib `io` before any zoo
# dataset/model loading that could transitively import `boto3`.
def _ensure_stdlib_io() -> None:
    """Restore the Python stdlib `io` module if plugin shadowing occurred."""
    import sys
    import sysconfig
    from importlib.util import module_from_spec, spec_from_file_location

    mod = sys.modules.get("io", None)
    if mod is not None and getattr(mod, "StringIO", None) is not None:
        return

    stdlib_dir = sysconfig.get_path("stdlib")
    if not stdlib_dir:
        # Best effort: if we can't locate stdlib, do nothing
        return

    io_path = os.path.join(stdlib_dir, "io.py")
    spec = spec_from_file_location("io", io_path)
    if spec is None or spec.loader is None:
        return

    stdlib_io = module_from_spec(spec)
    spec.loader.exec_module(stdlib_io)
    sys.modules["io"] = stdlib_io


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean-like environment variable."""
    value = os.environ.get(name, None)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _get_free_port(address: str) -> int:
    """Return an available TCP port for the provided address."""
    import socket

    info = socket.getaddrinfo(address, 0, socket.AF_UNSPEC, socket.SOCK_STREAM)
    family, socktype, proto, _, sockaddr = info[0]

    with socket.socket(family, socktype, proto) as s:
        s.bind(sockaddr)
        return int(s.getsockname()[1])


def _resolve_app_address(remote: bool) -> str:
    """Choose the app bind address from env or remote/local defaults."""
    address = os.environ.get("FIFTYONE_APP_ADDRESS", "").strip()
    if address:
        return address

    # Use an explicit IPv4 loopback to avoid "localhost" resolving to ::1,
    # which can cause the browser to connect to an older IPv6 server.
    if not remote:
        return "127.0.0.1"

    return "0.0.0.0"


def _is_fiftyone_server_running(address: str, port: int) -> bool:
    """Check whether a FiftyOne server responds at the given host and port."""
    import requests

    try:
        resp = requests.get(f"http://{address}:{port}/fiftyone", timeout=0.5)
        return resp.ok
    except Exception:
        return False


def _choose_port(address: str) -> int:
    """Choose an app port, avoiding an already-running FiftyOne server."""
    port_env = os.environ.get("FIFTYONE_APP_PORT", "").strip()
    if port_env:
        return int(port_env)

    for _ in range(10):
        port = _get_free_port(address)
        if not _is_fiftyone_server_running(address, port):
            return port

    return _get_free_port(address)


def _print_plugin_errors(*, plugin_name: str, raise_on_error: bool = True) -> None:
    """Prints any plugin registration errors with full tracebacks.

    Args:
        plugin_name: the plugin name, like "@51labs/few_shot_learning"
        raise_on_error: whether to raise if errors are found
    """
    import fiftyone.plugins.context as fopc

    ctxs = fopc.build_plugin_contexts(enabled=True)
    ctx_map = {c.name: c for c in ctxs}
    ctx = ctx_map.get(plugin_name)

    if ctx is None:
        print(f"\n[plugin-check] '{plugin_name}' was not discovered")
        print("[plugin-check] Discovered plugins:")
        for name in sorted(ctx_map.keys()):
            print(f"  - {name}")
        raise RuntimeError(f"Plugin not discovered: {plugin_name}")

    if not ctx.errors:
        return

    print(f"\n[plugin-check] '{plugin_name}' failed to register")
    for idx, err in enumerate(ctx.errors, 1):
        print(f"\n--- plugin error {idx}/{len(ctx.errors)} ---\n{err}")

    if raise_on_error:
        raise RuntimeError(
            f"Plugin '{plugin_name}' has registration errors; see tracebacks above"
        )


def _print_all_plugin_errors(*, raise_on_error: bool = False) -> None:
    """Prints full tracebacks for any plugins that failed to register."""
    import fiftyone.plugins.context as fopc

    ctxs = fopc.build_plugin_contexts(enabled=True)
    errored = [c for c in ctxs if getattr(c, "errors", None)]
    if not errored:
        return

    print("\n[plugin-check] Some plugins failed to register:")
    for ctx in sorted(errored, key=lambda c: c.name):
        print(f"\n== {ctx.name} ==")
        for idx, err in enumerate(ctx.errors, 1):
            print(f"\n--- plugin error {idx}/{len(ctx.errors)} ---\n{err}")

    if raise_on_error:
        names = ", ".join(sorted(c.name for c in errored))
        raise RuntimeError(f"Plugins with registration errors: {names}")


def main() -> None:
    """Run the demo workflow and launch the FiftyOne App."""
    # Load COCO subset (or use existing dataset)
    dataset_name = "few_shot_labs_demo"
    # Validate plugin registration and print full errors if it fails
    print("\nChecking plugin registration...")
    print(f"FIFTYONE_PLUGINS_DIR={os.environ.get('FIFTYONE_PLUGINS_DIR')}")
    print(f"fo.config.plugins_dir={fo.config.plugins_dir}")
    _print_all_plugin_errors(raise_on_error=False)
    _print_plugin_errors(plugin_name="@51labs/few_shot_learning")

    if dataset_name in fo.list_datasets():
        print(f"Loading existing dataset '{dataset_name}'...")
        dataset = fo.load_dataset(dataset_name)
    else:
        _ensure_stdlib_io()
        print("Loading COCO-2017 validation subset")
        dataset = foz.load_zoo_dataset(
            "coco-2017",
            split="validation",
            dataset_name=dataset_name,
        )

    print(f"Dataset: {dataset.name}")
    print(f"Samples: {len(dataset)}")

    # Compute embeddings if needed
    embedding_field = "resnet18_embeddings"
    schema = dataset.get_field_schema()

    if embedding_field not in schema:
        print(f"\nComputing {embedding_field}... (this may take a few minutes)")
        model = foz.load_zoo_model("resnet18-imagenet-torch")
        dataset.compute_embeddings(
            model,
            embeddings_field=embedding_field,
            batch_size=32,
            num_workers=4,
            skip_failures=True,
        )
        print(f"Embeddings computed and stored in '{embedding_field}'")
    else:
        print(f"Embeddings already exist in '{embedding_field}'")

    # Launch app
    print("\nLaunching FiftyOne App...")
    print("Open the 'Few-Shot Learning' panel from the + menu to begin!")
    print("\nWorkflow:")
    print("  1. Select model type and embedding field, then Start Session")
    print("  2. Select images and click 'Label Positive' or 'Label Negative'")
    print("  3. Click 'Train & Label Dataset' to predict on all samples")
    print("  4. Review predictions, add more labels, and iterate")
    print("  5. Click 'Tag Positives' to export your labeled samples")
    print("\nAvailable models:")
    print("  - RocchioPrototypeModel: Centroid-based prototype classifier (default)")

    # The default port (5151) is often already in use on dev machines (e.g. a
    # previous FiftyOne session). Pick a free port by default to avoid the
    # "Could not connect session..." loop.
    remote = _env_flag("FIFTYONE_APP_REMOTE", default=False)
    address = _resolve_app_address(remote)
    port = _choose_port(address)

    session = fo.launch_app(
        dataset,
        port=port,
        address=address,
        remote=remote,
        auto=not remote,
    )
    session.wait()


if __name__ == "__main__":
    main()
