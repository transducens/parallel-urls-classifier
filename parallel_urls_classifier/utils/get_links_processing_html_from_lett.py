
import sys
import base64
import logging
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from tldextract import extract
import html5lib # We won't use it, but we want an exception if can't import

def is_url_absolute(url):
    return bool(urlparse(url).netloc)

def main():
    print("src_lang\ttrg_lang\tsrc_url\ttrg_url\ttrg_tag\ttrg_url_original\tauthority_info") # Header

    for idx, lett_data in enumerate(sys.stdin):
        lett_data = lett_data.split('\t')
        lett_data[-1] = lett_data[-1].rstrip('\n')

        if len(lett_data) != 6:
            logging.error("Line %d: unexpected length: %d vs 6", idx + 1, len(lett_data))

            continue

        lang, mime, encoding, url, html_b64, text_b64 = lett_data
        url = url.replace('\t', ' ')

        html = base64.b64decode(html_b64).decode("utf-8", errors="ignore")

        if not html:
            logging.error("Line %d: could not get HTML", idx + 1)

            continue

        try:
            #parsed_html = BeautifulSoup(html, features="html.parser")
            parsed_html = BeautifulSoup(html, features="html5lib")
        except Exception as e:
            logging.error("Line %d: %s", idx + 1, str(e))

            continue

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
                    tag_url = tag_url.replace('\t', ' ')
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

                authority_info = "unk"

                try:
                    tsd, td, tsu = extract(url)
                    tag_tsd, tag_td, tag_tsu = extract(tag_url)

                    authority_info = f"{'equal' if tsd + '.' + td + '.' + tsu == tag_tsd + '.' + tag_td + '.' + tag_tsu else 'domain and TLD' if td + '.' + tsu == tag_td + '.' + tag_tsu else 'TLD' if tsu == tag_tsu else 'different'}"
                except Exception as e:
                    logging.warning("%s", str(e))

                print(f"{lang}\t{tag_lang}\t{url}\t{tag_url}\t{tag_name}\t{tag_original_url}\t{authority_info}")

if __name__ == "__main__":
    main()