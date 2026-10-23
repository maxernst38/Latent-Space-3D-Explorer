"""Serves the viewer against one or more local bundles.

A plain static file server on loopback: the built viewer out of the installed
package, the bundles out of wherever you built them. No application server,
because there is no work left to do at request time — `build` already did it.
"""

from __future__ import annotations

import functools
import http.server
import json
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import webbrowser
from pathlib import Path

VIEWER_DIR = Path(__file__).parent / "viewer"


def _is_wsl() -> bool:
    return "microsoft" in Path("/proc/version").read_text().lower() if (
        sys.platform == "linux" and Path("/proc/version").exists()
    ) else False


def open_browser(url: str) -> None:
    """Open `url`, handling WSL — where there is no Linux browser to open it with.

    Python's webbrowser module finds `gio` on a WSL box and hands the URL to it,
    which fails with "Operation not supported" because nothing inside the VM can
    display a window. The URL has to cross to Windows instead.
    """
    if _is_wsl():
        for command in (["wslview", url], ["explorer.exe", url]):
            try:
                subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
            except (FileNotFoundError, OSError):
                continue
        print("  (could not reach a Windows browser — open the URL above yourself)")
        return

    try:
        if not webbrowser.open(url):
            print("  (no browser found — open the URL above yourself)")
    except Exception:
        print("  (could not open a browser — open the URL above yourself)")


class _Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Bundles are regenerated in place; without this a re-run shows the
        # previous dataset from cache and looks like the export silently failed.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass  # The progress bars already said everything worth saying.


def serve(
    bundle_dirs: list[Path],
    host: str = "127.0.0.1",
    port: int = 8000,
    launch_browser: bool = True,
) -> None:
    if not VIEWER_DIR.is_dir():
        raise FileNotFoundError(
            f"No built viewer at {VIEWER_DIR}. Run `npm install && npm run build` "
            f"in web/ first."
        )

    root = _stage(bundle_dirs)
    handler = functools.partial(_Handler, directory=str(root))

    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer((host, port), handler) as httpd:
            url = f"http://{host}:{port}/"
            print(f"Viewer running at {url}  (ctrl-c to stop)")
            if launch_browser:
                threading.Timer(0.5, lambda: open_browser(url)).start()
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nStopped.")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _stage(bundle_dirs: list[Path]) -> Path:
    """Assemble a served tree in a temp dir: viewer assets plus the bundles.

    A scratch directory rather than serving VIEWER_DIR in place, because that
    directory lives inside the installed package — which is read-only under a
    system install, and shouldn't accumulate stray symlinks under any install.

    The entries are symlinks, not copies: bundles carry thumbnail atlases, and
    duplicating megabytes on every `view` would be waste with no upside.
    """
    root = Path(tempfile.mkdtemp(prefix="latent-explorer-"))

    for entry in VIEWER_DIR.iterdir():
        if entry.name != "bundles":
            (root / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())

    bundles_root = root / "bundles"
    bundles_root.mkdir()

    index = []
    for bundle_dir in bundle_dirs:
        bundle_dir = Path(bundle_dir).resolve()
        manifest_path = bundle_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"{bundle_dir} has no manifest.json.")

        manifest = json.loads(manifest_path.read_text())
        bundle_id = manifest.get("id", bundle_dir.name)
        (bundles_root / bundle_id).symlink_to(bundle_dir, target_is_directory=True)

        index.append(
            {
                "id": bundle_id,
                "name": manifest.get("name", bundle_id),
                "count": manifest.get("count", 0),
                "path": f"bundles/{bundle_id}",
            }
        )

    # The viewer reads this to populate its dataset picker.
    (bundles_root / "index.json").write_text(json.dumps(index, indent=2))
    return root
