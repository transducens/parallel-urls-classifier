
import re
import sys
import base64
import logging
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from tldextract import extract
import html5lib # We won't use it, but we want an exception if can't import

urls_to_process = sys.argv[1] if len(sys.argv) > 1 else None

def is_url_absolute(url):
    return bool(urlparse(url).netloc)

def main():
    # https://stackoverflow.com/questions/16549332/python-3-how-to-specify-stdin-encoding
    sys.stdin.reconfigure(encoding='utf-8', errors="backslashreplace")

    http_re_pattern = re.compile(r"^http")
    url_re_sub_blank = re.compile(r"\t|\r|\n")

    # Read urls to process
    urls2process = set()

    if urls_to_process is not None:
        with open(urls_to_process, 'r', encoding='utf-8', errors="backslashreplace") as f:
            for url in f:
                url = url_re_sub_blank.sub(' ', url).strip()

                urls2process.add(url)

        logging.info("urls2process: %d", len(urls2process))

    print("src_lang\ttrg_lang\tsrc_url\ttrg_url\ttrg_tag\ttrg_url_original\tauthority_info") # Header

    found_urls = set()

    for idx, lett_data in enumerate(sys.stdin):
        lett_data = lett_data.split('\t')
        lett_data[-1] = lett_data[-1].rstrip('\r\n')

        if len(lett_data) != 6:
            logging.error("Line %d: unexpected length: %d vs 6", idx + 1, len(lett_data))

            continue

        lang, mime, encoding, url, html_b64, text_b64 = lett_data
        url = url_re_sub_blank.sub(' ', url).strip()

        if len(urls2process) != 0:
            if url not in urls2process:
                continue

            found_urls.add(url)

        html = base64.b64decode(html_b64).decode("utf-8", errors="backslashreplace")

        if not html:
            logging.error("Line %d: could not get HTML", idx + 1)

            continue

        try:
            #parsed_html = BeautifulSoup(html, features="html.parser")
            parsed_html = BeautifulSoup(html, features="html5lib")
        except Exception as e:
            logging.error("Line %d: couldn't process the HTML: %s", idx + 1, str(e))

            continue

        lang_doc = None

        try:
            lang_doc = parsed_html.find("html")["lang"]
        except KeyError:
            pass
        except Exception as e:
            logging.warning("Line %d: couldn't find lang attr in html tag: %s", idx + 1, str(e))

        if lang_doc and lang_doc != lang:
            logging.warning("Line %d: document lang and provided lang are not the same: %s vs %s (using %s as lang)", idx + 1, lang_doc, lang, lang)

        link_tags = parsed_html.find_all("link")
        a_tags = parsed_html.find_all("a")

        for tags, tag_name in ((link_tags, "link"), (a_tags, "a")):
            for idx_tag, tag in enumerate(tags):
                try:
                    tag_url = tag["href"]
                    tag_url = url_re_sub_blank.sub(' ', tag_url).strip()
                except KeyError:
                    continue

                try:
                    tag_lang = tag["hreflang"]
                except KeyError:
                    tag_lang = "unk"

                tag_original_url = tag_url

                try:
                    if not is_url_absolute(tag_url):
                        # Resolve relative URL
                        tag_url = urljoin(url, tag_url)
                except ValueError:
                    pass

                if not http_re_pattern.match(tag_url):
                    logging.warning("Line %d: tag %d: tag url doesn't seem to match the pattern '^http': %s", idx, idx_tag, tag_url)

                    continue

                authority_info = "unk"

                try:
                    tsd, td, tsu = extract(url)
                    tag_tsd, tag_td, tag_tsu = extract(tag_url)

                    src_url_authority = '.'.join(part for part in [tsd, td, tsu] if part)
                    trg_url_authority = '.'.join(part for part in [tag_tsd, tag_td, tag_tsu] if part)
                    authority_info = "different"

                    if src_url_authority == trg_url_authority:
                        authority_info = "equal"
                    elif (td or tag_td) and f"{td}.{tsu}" == f"{tag_td}.{tag_tsu}":
                        authority_info = "domain and TLD"
                    elif tsu == tag_tsu:
                        authority_info = "TLD"
                except Exception as e:
                    logging.warning("%s", str(e))

                print(f"{lang}\t{tag_lang}\t{url}\t{tag_url}\t{tag_name}\t{tag_original_url}\t{authority_info}")

    if len(urls2process) != len(found_urls):
        logging.warning("%d found URLs were expected, but got %d", len(urls2process), len(found_urls))

        d = urls2process.difference(found_urls)

        for url in d:
            logging.warning("URL was expected to be found, but didn't: %s", url)

        d = found_urls.difference(urls2process)

        for url in d:
            logging.warning("Bug? URL: %s", url)

if __name__ == "__main__":
    main()