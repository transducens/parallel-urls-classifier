
import re
import sys
import base64
import logging
from urllib.parse import urljoin, urlparse

import utils

from bs4 import BeautifulSoup
from tldextract import extract
import html5lib # We won't use it, but we want an exception if can't import

logger = logging.getLogger("parallel_urls_classifier")
urls_to_process = sys.argv[1] if len(sys.argv) > 1 else None

def is_url_absolute(url):
    return bool(urlparse(url).netloc)

def main():
    # https://stackoverflow.com/questions/16549332/python-3-how-to-specify-stdin-encoding
    sys.stdin.reconfigure(encoding='utf-8', errors="backslashreplace")

    http_re_pattern = re.compile(r"^http")
    url_re_sub_blank = re.compile(r"\t|\r|\n")

    # Read urls to process (gold)
    urls2process = set()
    gs = set()
    gs_found = set()

    if urls_to_process is not None:
        logger.info("GS: expected format: english_url <tab> french_url")

        with open(urls_to_process, 'r', encoding='utf-8', errors="backslashreplace") as f:
            for pair in f:
                pair = pair.split('\t')

                if len(pair) != 2:
                    logger.warning("GS: 2 fields were expected, but got %d", len(pair))

                    continue

                gs_pair_src_url, gs_pair_trg_url = pair

                gs_pair_src_url = url_re_sub_blank.sub(' ', gs_pair_src_url).strip()
                gs_pair_trg_url = url_re_sub_blank.sub(' ', gs_pair_trg_url).strip()

                urls2process.add(gs_pair_src_url)
                urls2process.add(gs_pair_trg_url)
                gs.add(f"en\t{gs_pair_src_url}\t{gs_pair_trg_url}") # english side
                gs.add(f"fr\t{gs_pair_trg_url}\t{gs_pair_src_url}") # french side

        logger.info("urls2process: %d", len(urls2process))

    print("src_lang\ttrg_lang\tsrc_url\ttrg_url\ttrg_tag\ttrg_url_original\tauthority_info\tgs_info\ttag_url_found_in_dataset") # Header

    found_urls = set()
    final_entries = []
    urls2lang = {}

    for idx, lett_data in enumerate(sys.stdin):
        lett_data = lett_data.split('\t')
        lett_data[-1] = lett_data[-1].rstrip('\r\n')

        if len(lett_data) != 6:
            logger.error("Line %d: unexpected length: %d vs 6", idx + 1, len(lett_data))

            continue

        lang, mime, encoding, url, html_b64, text_b64 = lett_data
        url = url_re_sub_blank.sub(' ', url).strip()

        if url not in urls2lang:
            urls2lang[url] = set()

        urls2lang[url].add(lang)

        if len(urls2process) != 0:
            if url not in urls2process:
                continue

            found_urls.add(url)

        html = base64.b64decode(html_b64).decode("utf-8", errors="backslashreplace")

        if not html:
            logger.error("Line %d: could not get HTML", idx + 1)

            continue

        try:
            #parsed_html = BeautifulSoup(html, features="html.parser")
            parsed_html = BeautifulSoup(html, features="html5lib")
        except Exception as e:
            logger.error("Line %d: couldn't process the HTML: %s", idx + 1, str(e))

            continue

        lang_doc = None

        try:
            lang_doc = parsed_html.find("html")["lang"]
        except KeyError:
            pass
        except Exception as e:
            logger.warning("Line %d: couldn't find lang attr in html tag: %s", idx + 1, str(e))

        if lang_doc and lang_doc != lang:
            logger.warning("Line %d: document lang and provided lang are not the same: %s vs %s (using %s as lang)", idx + 1, lang_doc, lang, lang)

        link_tags = parsed_html.find_all("link")
        a_tags = parsed_html.find_all("a")
        print_data = []
        gs_info_found = False
        seen_trg_urls = set()

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
                    logger.warning("Line %d: tag %d: tag url doesn't seem to match the pattern '^http': %s", idx, idx_tag, tag_url)

                    continue

                authority_info = "unk"
                gs_info = "none"

                if len(gs) != 0:
                    gs_pair = f"{lang}\t{url}\t{tag_url}"

                    if gs_pair in gs:
                        gs_info = "yes"
                        gs_info_found = True

                        gs_found.add(gs_pair)
                    else:
                        # Some websites seem to remove dinamically the extension (e.g. 1d-aquitaine.com), and
                        #  the provided URLs seem to be from the dynamic content while the HTML don't
                        #  e.g.: http://1d-aquitaine.com/en/artist/ex.html in the HTML but http://1d-aquitaine.com/en/artist/ex in the GS

                        if tag_url.endswith(".html"):
                            gs_pair = f"{lang}\t{url}\t{tag_url[:-5]}"

                            if gs_pair in gs:
                                gs_info = "almost"
                                gs_info_found = True

                                gs_found.add(gs_pair)

                        if tag_url.endswith(".htm"):
                            gs_pair = f"{lang}\t{url}\t{tag_url[:-4]}"

                            if gs_pair in gs:
                                gs_info = "almost"
                                gs_info_found = True

                                gs_found.add(gs_pair)

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
                    logger.warning("%s", str(e))

                entry = f"{lang}\t{tag_lang}\t{url}\t{tag_url}\t{tag_name}\t{tag_original_url}\t{authority_info}\t{gs_info}"

                if tag_url in seen_trg_urls:
                    continue

                seen_trg_urls.add(tag_url)
                print_data.append(entry)

        if len(gs) == 0 or gs_info_found:
            final_entries.extend(print_data)
        else:
            logger.warning("Couldn't find the GS pair for the processed URL (lang: %s): %s", lang, url)

    for entry in final_entries:
        entry_parts = entry.split('\t')
        tag_url = entry_parts[3]
        url_found_info = "not_in_data"
        tag_url_processed = None

        if tag_url in urls2lang:
            tag_url_processed = ','.join(sorted(list(urls2lang[tag_url])))
        else:
            if tag_url.endswith(".html") and tag_url[:-5] in urls2lang:
                tag_url_processed = f"almost: {','.join(sorted(list(urls2lang[tag_url[:-5]])))}"
            if tag_url.endswith(".htm") and tag_url[:-4] in urls2lang:
                tag_url_processed = f"almost: {','.join(sorted(list(urls2lang[tag_url[:-4]])))}"

        if tag_url_processed:
            url_found_info = f"found_langs: {tag_url_processed}"

        entry += f"\t{url_found_info}"

        print(entry)

    if len(urls2process) != len(found_urls):
        logger.warning("%d found URLs were expected, but got %d", len(urls2process), len(found_urls))

        d = urls2process.difference(found_urls)

        for url in d:
            logger.warning("URL was expected to be found, but didn't: %s", url)

        d = found_urls.difference(urls2process)

        for url in d:
            logger.error("URL was not expected to be found, but did: bug?: %s", url)

    if len(gs) != len(gs_found):
        logger.warning("GS: %d found URLs were expected, but got %d", len(gs), len(gs_found))

        d = gs.difference(gs_found)

        for url in d:
            logger.warning("GS: URL was expected to be found, but didn't: %s", url)

        d = gs_found.difference(gs)

        for url in d:
            logger.error("GS: URL was not expected to be found, but did: bug?: %s", url)

if __name__ == "__main__":
    logger = utils.set_up_logging_logger(logger, level=logging.DEBUG)

    main()