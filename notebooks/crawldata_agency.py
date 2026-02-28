#!/usr/bin/env python3
import argparse
import os
import re
import time
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

# --- CONSTANTS ---
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


# --- GROUP: URL HANDLING ---
def canonicalize_url(u: str) -> str:
    p = urlparse(u.strip())
    p = p._replace(fragment="", query="")
    return urlunparse(p)


def is_profile_url(u: str) -> bool:
    # Ví dụ: https://www.agencyvietnam.com/profile/5531-unique-ooh
    p = urlparse(u)
    return (p.netloc in {"www.agencyvietnam.com", "agencyvietnam.com"}) and p.path.startswith("/profile/")


def safe_slug(url: str) -> str:
    p = urlparse(url)
    name = p.path.strip("/").replace("/", "_") or "page"
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name[:150]


# --- GROUP: REQUEST & EXTRACTION ---
def fetch(session: requests.Session, url: str, timeout: int) -> requests.Response:
    r = session.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r


def extract_profile_links(html: str, base_url: str, limit: int) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    seen = set()
    out = []

    for a in soup.select("a[href]"):
        u = canonicalize_url(urljoin(base_url, a.get("href", "")))
        if is_profile_url(u) and u not in seen:
            seen.add(u)
            out.append(u)
            if len(out) >= limit:
                break

    return out


# --- MAIN EXECUTION ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-url", default="https://www.agencyvietnam.com/")
    ap.add_argument("--max", type=int, default=10)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--timeout", type=int, default=25)
    ap.add_argument("--out-dir", default="out/agencyvietnam")
    ap.add_argument("--out-urls", default="out/agencyvietnam/urls.txt")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with requests.Session() as s:
        # 1) Tải raw index  
        index_url = canonicalize_url(args.index_url)
        r = fetch(s, index_url, args.timeout)

        index_path = os.path.join(args.out_dir, "index.html")
        with open(index_path, "wb") as f:
            f.write(r.content)

        # 2) Extract profile links
        profile_urls = extract_profile_links(r.text, index_url, args.max)

        with open(args.out_urls, "w", encoding="utf-8") as f:
            f.write("\n".join(profile_urls) + "\n")

        # 3) Crawl từng profile -> lưu raw html
        for i, u in enumerate(profile_urls, 1):
            time.sleep(args.delay)
            try:
                rr = fetch(s, u, args.timeout)
                out_path = os.path.join(args.out_dir, f"{i:03d}_{safe_slug(u)}.html")
                with open(out_path, "wb") as f:
                    f.write(rr.content)
                print(f"[{i}/{len(profile_urls)}] OK  {out_path}")
            except Exception as e:
                print(f"[{i}/{len(profile_urls)}] FAIL {u} -> {e}")


if __name__ == "__main__":
    main()
