---
source: Context7 API
library: Playwright for Python
package: playwright-python
topic: page.goto response/status/redirect/block-page guards for RAG ingestion
fetched: 2026-04-22T00:00:00Z
official_docs: https://playwright.dev/python/docs/api/class-page
---

## Key API behaviors to build guards around

- `page.goto(url, wait_until=...)` returns the **main resource response** for navigation.
- Playwright throws on navigation failures like SSL errors, invalid URL, or unreachable host.
- Playwright **does not throw** for HTTP status errors (e.g., 404/500); these must be checked explicitly.
- Request lifecycle emits: `request` -> `response` -> `requestfinished` (or `requestfailed` on network error).
- Redirects create a new request; you can inspect chains via `request.redirected_from` / `request.redirected_to`.

## Practical ingestion quality gates (Python)

```python
from playwright.sync_api import sync_playwright

BLOCK_HINTS = [
    "captcha", "access denied", "forbidden", "verify you are human",
    "cloudflare", "bot detection", "temporarily blocked"
]

def get_redirect_chain(req):
    chain = []
    cur = req
    while cur is not None:
        chain.append(cur.url)
        cur = cur.redirected_from
    return list(reversed(chain))

def fetch_page_guarded(url: str):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        failed_requests = []
        page.on("requestfailed", lambda r: failed_requests.append({
            "url": r.url,
            "method": r.method,
            "error": r.failure.error_text if r.failure else "unknown"
        }))

        try:
            resp = page.goto(url, wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            browser.close()
            return {"ok": False, "reason": f"navigation_error:{e}"}

        if resp is None:
            browser.close()
            return {"ok": False, "reason": "no_main_response"}

        status = resp.status
        final_url = page.url
        chain = get_redirect_chain(resp.request)

        # Gate 1: HTTP status
        if status < 200 or status >= 300:
            browser.close()
            return {"ok": False, "reason": f"bad_status:{status}", "final_url": final_url, "redirect_chain": chain}

        # Gate 2: redirect/domain sanity
        if len(chain) > 5:
            browser.close()
            return {"ok": False, "reason": "redirect_chain_too_long", "redirect_chain": chain}

        # Gate 3: blocked/captcha heuristics
        text = page.content().lower()
        if any(h in text for h in BLOCK_HINTS):
            browser.close()
            return {"ok": False, "reason": "blocked_or_captcha_page", "final_url": final_url}

        # Gate 4: failed critical requests (optional stricter policy)
        if failed_requests:
            # keep/deny based on your policy; strict pipelines should deny
            browser.close()
            return {"ok": False, "reason": "request_failures_detected", "failures": failed_requests[:5]}

        html = page.content()
        browser.close()
        return {
            "ok": True,
            "status": status,
            "final_url": final_url,
            "redirect_chain": chain,
            "html": html,
        }
```

## Recommended defaults

- Use `wait_until="domcontentloaded"` for ingestion stability (avoid over-waiting pitfalls).
- Always persist `status`, `final_url`, and `redirect_chain` in ingestion metadata/logs.
- Quarantine (don’t embed) pages that fail any guard; send to retry/review queue.

## Sources used

- https://playwright.dev/python/docs/api/class-page
- https://playwright.dev/python/docs/api/class-request
- https://playwright.dev/python/docs/api/class-response
- https://playwright.dev/python/docs/network
