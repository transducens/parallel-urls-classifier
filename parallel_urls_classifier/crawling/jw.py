
import os
import sys
import pathlib
from datetime import datetime
import urllib.request
import multiprocessing
import time
import socket
import concurrent.futures
import http.client
import urllib.error

import xmltodict
from bs4 import BeautifulSoup

def download_file(_url, _timeout=30.0, force=True):
#    user_agent = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:97.0) Gecko/20100101 Firefox/97.0"
#    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36"
#    user_agent = "PostmanRuntime/7.6.0"

    try:
        with urllib.request.urlopen(_url, timeout=_timeout) as f:
            return f.read().decode("utf-8")

#        req = urllib.request.Request(_url, data=None, headers={'User-Agent': user_agent})

#        return urllib.request.urlopen(req, timeout=_timeout).read().decode("utf-8")
    except socket.timeout as e:
        sys.stderr.write(f"Coult not download '{_url}' (timeout): {e}\n")

        return None
    except http.client.IncompleteRead as e:
        sys.stderr.write(f"Coult not download '{_url}' (incomplete read): {e}\n")

        return None
    except urllib.error.HTTPError as e:
        sys.stderr.write(f"Coult not download '{_url}' (HTTP error): {e}\n")

        if force:
            sys.stderr.write("Waiting...\n")
            time.sleep(60 * 2)

            return download_file(_url, _timeout, False)

        return None
    except urllib.error.URLError as e:
        sys.stderr.write(f"Coult not download '{_url}' (URL error): {e}\n")

        if force:
            sys.stderr.write("Waiting...\n")
            time.sleep(60 * 2)

            return download_file(_url, _timeout, False)

        return None
    except Exception as e:
        sys.stderr.write(f"Coult not download '{_url}' (exception): {e}\n")

        return None


output_prefix = pathlib.Path(__file__).parent.resolve()
output_dir = datetime.now().strftime("%Y%m%d%H%M%S")
output = f"{output_prefix}/{output_dir}"
langs = sys.argv[1:]
sitemap_en_link = "https://www.jw.org/en/sitemap.xml"

#os.mkdir(output)

#print(f"Output: {output}")

sitemap = download_file(sitemap_en_link, 120.0)

if sitemap is None:
    sys.stderr.write("Could not download the sitemap.xml\n")
    sys.exit(1)

#with open(f"{output}/sitemap-en.xml", "w") as f:
#    f.write(sitemap)

sitemap_dict = xmltodict.parse(sitemap)

#print(sitemap_dict["urlset"]["url"])

def process(args):
    _loc, _wait = args
    results = []

    time.sleep(_wait)

    try:
        html = download_file(_loc)

        if not html:
            return results

        parsed_html = BeautifulSoup(html, features="html.parser")
        links = parsed_html.head.find_all("link")

        for link in links:
            try:
                lang = link["hreflang"]
                href = link["href"]

                if lang in langs:
                    en_url = _loc.replace('\t', ' ')
                    trg_url = href.replace('\t', ' ')

                    results.append(f"en\t{lang}\t{en_url}\t{trg_url}")
            except KeyError:
                continue

        return results
    except Exception as e:
        sys.stderr.write("Error process(): " + str(e) + '\n')

        return results

if __name__ == '__main__':
    locs = []
    noprocesses = 16
    #pool = multiprocessing.Pool(processes=noprocesses)

    multiprocessing.set_start_method("spawn")

    # https://bugs.python.org/issue38744
#    with multiprocessing.get_context("spawn").Pool(processes=noprocesses) as pool:
    # https://stackoverflow.com/questions/65115092/occasional-deadlock-in-multiprocessing-pool
    with concurrent.futures.ProcessPoolExecutor() as pool:
        for url in sitemap_dict["urlset"]["url"]:
            loc = url["loc"]

            if len(locs) < noprocesses:
                locs.append((loc, len(locs) / 10 * 3))
            else:
#                result = pool.starmap(process, locs)
                result = pool.map(process, locs)

                time.sleep(5)

                for r in result:
                    for u in r:
                        print(u)

                locs = []

        if len(locs) != 0:
            result = pool.map(process, locs)

            for r in result:
                for u in r:
                    print(u)
