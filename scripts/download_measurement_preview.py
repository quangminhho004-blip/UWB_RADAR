"""Download three public development recordings using HTTP byte ranges."""
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile

URL = "https://zenodo.org/records/15022885/files/tripod.zip"
ROOT = Path(__file__).resolve().parents[1]

class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.position = 0
        request = urllib.request.Request(URL, headers={"Range": "bytes=-22"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206:
                raise RuntimeError("Server does not support partial downloads.")
            self.size = int(response.headers["Content-Range"].split("/")[-1])

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size <= 0:
            return b""
        start, end = self.position, self.position + size - 1
        if size > 32 * 1024 * 1024:
            raise RuntimeError("Unexpectedly large range request.")
        request = urllib.request.Request(URL + f"?preview_range={start}-{end}",
                                         headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=90) as response:
            expected = f"bytes {start}-{end}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise RuntimeError("Server returned a different byte range.")
            data = response.read(size + 1)
        if len(data) != size:
            raise RuntimeError("Truncated byte range.")
        self.position += size
        return data

def main():
    target = ROOT / "data/measurement_preview"
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(RemoteZip()) as archive:
        names = sorted(archive.namelist())
        for user in "ACF":
            candidates = [n for n in names if re.search(r"user" + user + r"_", n) and n.endswith(".csv")]
            if not candidates:
                raise RuntimeError("No recording for subject " + user)
            for member in candidates:
                path = target / Path(member).name
                content = archive.read(member)  # zipfile verifies the member CRC.
                records = list(csv.reader(io.StringIO(content.decode("utf-8"))))
                if len(records) == 1500 and all(len(r) >= 254 for r in records):
                    path.write_bytes(content)
                    break
                print("Skipping incomplete recording:", Path(member).name, len(records), flush=True)
            else:
                raise RuntimeError("No complete recording for subject " + user)
            rows.append(dict(user=user, archive_member=member, local_file=path.name,
                             source_url=URL, sha256=hashlib.sha256(content).hexdigest(),
                             bytes=len(content)))
            print("Downloaded:", path.name, len(content), flush=True)
    (target / "provenance.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
