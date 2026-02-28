#!/usr/bin/env python3
import argparse
import os
import re
import time
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

# --- CONSTANTS & CONFIG ---
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

ARTICLE_SELECTORS = [
    "article.fck_detail",
    "div.fck_detail",
    "article",
    "div.detail-content",
    "div.content_detail",
]

TITLE_SELECTORS = [
    "h1.title-detail",
    "h1",
]

LISTING_PREFIX = "https://vnexpress.net/du-lich/diem-den/viet-nam"


# --- GROUP: URL HANDLING & FILTERING ---
def canonicalize_url(u: str) -> str:
    u = u.strip()
    if not u:
        return ""
    p = urlparse(u)
    p = p._replace(fragment="", query="")
    return urlunparse(p)


def is_article_url(u: str) -> bool:
    # VnExpress bài thường kết thúc .html; lọc bớt link không phải bài.
    return u.startswith("https://vnexpress.net/") and u.endswith(".html")


def is_listing_url(u: str) -> bool:
    # Trang chuyên mục/địa phương: cùng prefix, không phải bài .html
    return u.startswith(LISTING_PREFIX) and not u.endswith(".html")


def safe_slug(url: str) -> str:
    path = urlparse(url).path
    name = path.rsplit("/", 1)[-1] or "article"
    name = name.replace(".html", "")
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name[:120] or "article"


# --- GROUP: NETWORKING & COLLECTION ---
def fetch_html(session: requests.Session, url: str, timeout: int = 20) -> str:
    r = session.get(url, timeout=timeout, headers=DEFAULT_HEADERS)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or r.encoding
    return r.text


def collect_article_urls(session: requests.Session, start_url: str, max_urls: int, max_pages: int = 50) -> list[str]:
    seen_articles = set()
    seen_pages = set()
    queue = [start_url]
    out = []

    while queue and len(out) < max_urls and len(seen_pages) < max_pages:
        page_url = queue.pop(0)
        page_url = canonicalize_url(page_url)
        if page_url in seen_pages:
            continue
        seen_pages.add(page_url)

        try:
            html = fetch_html(session, page_url)
            soup = BeautifulSoup(html, "lxml")

            for a in soup.select("a[href]"):
                href = a.get("href", "")
                u = canonicalize_url(urljoin(page_url, href))

                if is_article_url(u) and u not in seen_articles:
                    seen_articles.add(u)
                    out.append(u)
                    if len(out) >= max_urls:
                        break

                # Nếu gặp trang con (listing) thì cho vào hàng đợi để crawl tiếp
                if is_listing_url(u) and u not in seen_pages and u not in queue:
                    queue.append(u)
        except Exception as e:
            print(f"Skipping listing {page_url} due to error: {e}")

    return out


# --- GROUP: CONTENT EXTRACTION ---
def pick_first(soup: BeautifulSoup, selectors: list[str]):
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            return el
    return None


def extract_article_text(article_html: str) -> tuple[str, str]:
    soup = BeautifulSoup(article_html, "lxml")

    title_el = pick_first(soup, TITLE_SELECTORS)
    title = title_el.get_text(" ", strip=True) if title_el else ""

    container = pick_first(soup, ARTICLE_SELECTORS)
    if not container:
        return title, ""

    # Bỏ các tag rác thường gặp
    for t in container.select("script, style, noscript, iframe, svg"):
        t.decompose()

    # Chỉ lấy block chính: p + heading + caption (bỏ li để tránh trùng do box/list)
    blocks = container.select("h2, h3, p, figcaption")

    texts = []
    last = None
    for el in blocks:
        txt = el.get_text(" ", strip=True)
        txt = re.sub(r"\s+", " ", txt).strip()
        if not txt:
            continue

        # Dedupe theo thứ tự (trùng y hệt liên tiếp)
        if txt == last:
            continue
        last = txt
        texts.append(txt)

    # Cắt dòng chữ ký tác giả ở cuối (thường là 1 tên ngắn)
    if texts and re.fullmatch(r"[A-Za-zÀ-Ỵà-ỵ .'-]{2,40}", texts[-1]):
        # ví dụ "Tâm Anh"
        texts.pop()

    body = "\n\n".join(texts).strip()
    return title, body


# --- MAIN EXECUTION ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-url", required=True)
    ap.add_argument("--max", type=int, default=100)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--out-urls", default="out/travel/urls.txt")
    ap.add_argument("--out-dir", default="out/travel")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with requests.Session() as s:
        # (1) Thu thập URL bài viết bằng BFS qua các trang listing
        urls = collect_article_urls(s, args.index_url, args.max)

        with open(args.out_urls, "w", encoding="utf-8") as f:
            for u in urls:
                f.write(u + "\n")

        # (2) Duyệt từng URL, lấy nội dung
        for i, u in enumerate(urls, 1):
            time.sleep(args.delay)
            try:
                html = fetch_html(s, u)
                title, body = extract_article_text(html)

                slug = safe_slug(u)
                out_path = os.path.join(args.out_dir, f"{i:03d}_{slug}.txt")
                with open(out_path, "w", encoding="utf-8") as f:
                    if title:
                        f.write(title.strip() + "\n\n")
                    f.write(u + "\n\n")
                    f.write(body.strip() + "\n")
                print(f"[{i}/{len(urls)}] OK  {out_path}")
            except Exception as e:
                print(f"[{i}/{len(urls)}] FAIL {u} -> {e}")


if __name__ == "__main__":
    main()