
import sys
import base64
import logging
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

def is_url_absolute(url):
    return bool(urlparse(url).netloc)

def main():
    print("src_lang\ttrg_lang\tsrc_url\ttrg_url\ttrg_tag\ttrg_url_original") # Header

    for idx, lett_data in enumerate(sys.stdin):
        lett_data = lett_data.strip().split('\t')
        urls = []
        langs = []

        if len(lett_data) != 6:
            raise Exception(f"Unexpected length: {len(lett_data)} vs 6")

        lang, mime, encoding, url, html_b64, text_b64 = lett_data

        html = base64.b64decode(html_b64).decode("utf-8", errors="ignore")

        if not html:
            logging.warning("Line %d: could not get HTML", idx + 1)

            continue

        parsed_html = BeautifulSoup(html, features="html.parser")
        lang_doc = None

        try:
            lang_doc = parsed_html.find("html")["lang"]
        except KeyError:
            pass
        except Exception as e:
            logging.warning("Line %d: %s", idx + 1, str(e))

        if lang_doc and lang_doc != lang:
            logging.warning("Line %d: document lang and provided lang are not the same: %s vs %s (using %s as lang)", idx + 1, lang_doc, lang, lang)

        link_tags = parsed_html.find_all("link")
        a_tags = parsed_html.find_all("a")

        for tags, tag_name in ((link_tags, "link"), (a_tags, "a")):
            for tag in tags:
                try:
                    tag_url = tag["href"]
                except KeyError:
                    continue

                try:
                    tag_lang = tag["hreflang"]
                except KeyError:
                    tag_lang = "unk"

                tag_original_url = tag_url

                if not is_url_absolute(tag_url):
                    # Resolve relative URL
                    tag_url = urljoin(url, tag_url)

                print(f"{lang}\t{tag_lang}\t{url}\t{tag_url}\t{tag_name}\t{tag_original_url}")

if __name__ == "__main__":
    main()