
import sys
import pathlib
from datetime import datetime
import multiprocessing
import time
import concurrent.futures
import base64

from bs4 import BeautifulSoup

output_prefix = pathlib.Path(__file__).parent.resolve()
output_dir = datetime.now().strftime("%Y%m%d%H%M%S")
output = f"{output_prefix}/{output_dir}"
langs = sys.argv[1:]

def process(base64_data):
    en_url, base64_html = base64_data.split('\t')
    results = []

    try:
        html = base64.b64decode(base64_html).decode("utf-8", errors="ignore")

        if not html:
            return results

        parsed_html = BeautifulSoup(html, features="html.parser")
        html_tag = parsed_html.find("html")
        links = parsed_html.head.find_all("link")

        for link in links:
            try:
                lang = link["hreflang"]
                href = link["href"]

                try:
                    src_lang = html_tag["lang"]
                except KeyError:
                    src_lang = "unk"

                if lang in langs or '*' in langs:
                    trg_url = href.replace('\t', ' ')

                    results.append(f"{src_lang}\t{lang}\t{en_url}\t{trg_url}")
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
        for base64_data in sys.stdin:
            if len(locs) < noprocesses:
                locs.append(base64_data)
            else:
#                result = pool.starmap(process, locs)
                result = pool.map(process, locs)

                for r in result:
                    for u in r:
                        print(u)

                locs = []

        if len(locs) != 0:
            result = pool.map(process, locs)

            for r in result:
                for u in r:
                    print(u)
