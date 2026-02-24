import time, requests, urllib3
import xml.etree.ElementTree as ET
from fastapi.responses import PlainTextResponse
from fastapi import FastAPI, Form
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from lxml import html

app = FastAPI()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================= PROXY CONFIG =================
PROXY = "http://172.16.180.43:80"
USE_PROXY = True
# =================================================

HEADERS = {
    "User-Agent": "Chrome/121.0.0.0 Safari/537.36",
    "Accept-Language": "en-IN,en;q=0.9",
}


# =================================================
# SESSION CREATOR (scraper.py style)
# =================================================
def create_session(use_proxy=True) -> requests.Session:
    session = requests.Session()

    # Prevent environment proxy override
    session.trust_env = False

    retry = Retry(
        total=1,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=10,
        pool_maxsize=20
    )

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.headers.update(HEADERS)

    if use_proxy and PROXY:
        session.proxies = {
            "http": PROXY,
            "https": PROXY,
        }

    return session


# =================================================
# FETCH TEXT (proxy + automatic fallback)
# =================================================
def fetch_text(session: requests.Session, url: str, timeout: int = 30, retries: int = 3) -> str:
    global USE_PROXY
    last_err = None

    for attempt in range(retries):
        try:
            with create_session(use_proxy=USE_PROXY) as s:
                resp = s.get(
                    url,
                    timeout=timeout,
                    verify=False,
                    allow_redirects=True
                )

                resp.raise_for_status()
                return resp.text

        except requests.exceptions.ProxyError as e:
            print(f"[PROXY ERROR] {e}")
            print("[FALLBACK] Switching to direct connection...")
            USE_PROXY = False
            last_err = e

        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Failed to fetch {url} after {retries} retries. Last error: {last_err}")


# =================================================
# PARSE SITEMAP (UNCHANGED)
# =================================================
def parse_sitemap(xml_text: str):
    root = ET.fromstring(xml_text)

    if "}" in root.tag:
        ns_uri = root.tag.split("}")[0].strip("{")
        ns = {"sm": ns_uri}
        loc_elems = root.findall(".//sm:loc", ns)
    else:
        loc_elems = root.findall(".//loc")

    locs = [(e.text or "").strip() for e in loc_elems if (e.text or "").strip()]
    kind = root.tag.split("}")[-1]
    return kind, locs


# =================================================
# FIND URL (UNCHANGED LOGIC)
# =================================================
def find_urls_by_keyword(keyword: str, max_sub_sitemaps: int = 250, polite_sleep: float = 0.2):

    keyword = keyword.lower()
    sitemap_url = "https://www.chittorgarh.com/sitemap.xml"

    matches = []
    scanned = 0

    with requests.Session() as session:

        main_xml = fetch_text(session, sitemap_url)
        kind, locs = parse_sitemap(main_xml)

        if kind == "sitemapindex":
            for sm_url in locs:
                scanned += 1
                if scanned > max_sub_sitemaps:
                    break

                time.sleep(polite_sleep)

                try:
                    sub_xml = fetch_text(session, sm_url)
                    _, sub_locs = parse_sitemap(sub_xml)

                    for u in sub_locs:
                        if keyword in u.lower():
                            matches.append(u)

                except Exception as e:
                    print(f"[WARN] Could not read sub-sitemap: {sm_url} -> {e}")

        else:
            for u in locs:
                if keyword in u.lower():
                    matches.append(u)

    seen = set()
    uniq = []
    for u in matches:
        if u not in seen:
            seen.add(u)
            uniq.append(u)

    return uniq


# =================================================
# GET URL (UNCHANGED)
# =================================================
def get_url(query):
    if "limited" in query.lower():
        queryx = query.lower()
        q = queryx.replace("limited", "")
    else:
        q = query.lower()

    q = q.strip().replace(" ", "-")
    keyword = f"ipo/{q}"

    results = find_urls_by_keyword(keyword)

    if results:
        return results[0]
    else:
        return "Not found url"


# =================================================
# GET DATA (SCRAPER LOGIC UNCHANGED)
# =================================================
def get_data_xml(url):

    global USE_PROXY

    try:
        with create_session(use_proxy=USE_PROXY) as session:
            response = session.get(url, verify=False)

            response.raise_for_status()

    except requests.exceptions.ProxyError as e:
        print(f"[PROXY ERROR] {e}")
        print("[FALLBACK] Switching to direct connection...")
        USE_PROXY = False

        with create_session(use_proxy=False) as session:
            response = session.get(url, verify=False)
            response.raise_for_status()

    tree = html.fromstring(response.content)

    divs = tree.xpath("//div[contains(@class,'card') and contains(@class,'p-3')]")
    accept_div = []
    jcontent = ""

    for div in divs:
        tbls = div.find(".//table")
        lst = div.find(".//ul")

        if (tbls or lst) is not None:
            accept_div.append(div)

    for div in accept_div:
        jcontent += div.text_content().strip()
        jcontent += "\n\n"

    return jcontent
