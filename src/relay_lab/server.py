"""Local replay hosting with single HTTP byte ranges for seekable MP4 video."""

import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class ReplayRequestHandler(SimpleHTTPRequestHandler):
    """Serve static replay files, including bounded/open-ended/suffix ranges.

    Multiple ranges and unknown range units fall back to a normal full response.
    The server does not emit entity tags; an If-Range request conservatively gets
    the full representation rather than risking a partial response for stale data.
    """

    protocol_version = "HTTP/1.1"

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        self._byte_range = None
        requested = self.headers.get("Range", "")
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip(), flags=re.IGNORECASE)
        path = Path(self.translate_path(self.path))
        if not match or self.headers.get("If-Range") or not path.is_file():
            return super().send_head()
        try:
            stream = path.open("rb")
        except OSError:
            self.send_error(404, "File not found")
            return None
        try:
            stat = os.fstat(stream.fileno())
            size = stat.st_size
            left, right = match.groups()
            if left:
                start = int(left)
                end = min(int(right), size - 1) if right else size - 1
            elif right and int(right) > 0:
                start, end = max(0, size - int(right)), size - 1
            else:
                raise ValueError("Empty byte range")
            if start >= size or start > end:
                raise ValueError("Unsatisfiable byte range")
        except ValueError:
            stream.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        except OSError:
            stream.close()
            self.send_error(404, "File not found")
            return None
        self._byte_range = (start, end)
        stream.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
        self.end_headers()
        return stream

    def copyfile(self, source, outputfile):
        if self._byte_range is None:
            return super().copyfile(source, outputfile)
        start, end = self._byte_range
        remaining = end - start + 1
        while remaining:
            chunk = source.read(min(remaining, 64 * 1024))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def serve(directory: Path, port: int = 8766) -> None:
    """Host a directory on loopback until Ctrl-C, without changing working directory."""
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Replay directory does not exist: {directory}")
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    handler = partial(ReplayRequestHandler, directory=str(directory))
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        print(f"Replay server: http://127.0.0.1:{server.server_port}/", flush=True)
        print(f"Directory: {directory}\nPress Ctrl-C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nReplay server stopped.")
