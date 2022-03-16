
import sys
import requests
import os
import logging
from contextlib import contextmanager
import json
import gzip
import lzma

@contextmanager
def open_xz_or_gzip_or_plain(file_path, mode='rt'):
    f = None
    try:
        if file_path[-3:] == ".gz":
            f = gzip.open(file_path, mode)
        elif file_path[-3:] == ".xz":
            f = lzma.open(file_path, mode)
        else:
            f = open(file_path, mode)
        yield f

    except Exception:
        raise Exception("Error occurred while loading a file!")

    finally:
        if f:
            f.close()

prefixes = [i.strip() for i in sys.stdin]
url_base = "https://commoncrawl.s3.amazonaws.com"

for prefix in prefixes:
    logging.info("Processing prefix '%s'", prefix)

    filename = prefix.split('/')[-1]

    # Download file
    url = f"{url_base}/{prefix}"
    r = requests.get(url, allow_redirects=True)
    content = r.content

    # Store file
    open(filename, 'wb').write(content)

    # Process
    with open_xz_or_gzip_or_plain(filename) as f:
        for line in f:
            line = line.strip()
            line = line[line.find(" {\"url\": "):]
            line = json.loads(line)

            print(line["url"])

    # Remove
    os.system(f"rm -rf {filename}")
