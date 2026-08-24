"""Drive the browser demo headlessly and screenshot it — the only way to catch a layout bug.

The engine can be byte-identical to the native build and the page can still draw the boxes in the
wrong place; `wasm/test_node.js` cannot see that, because it never lays anything out. This serves
the repo over HTTP, loads `wasm/`, clicks the sample image, reads it, and saves a screenshot plus
the box coordinates it read back out of the page.

  python -m playwright install chromium      # once
  python tools/wasm_smoke.py --out scratch/wasm_page.png
"""
import argparse
import http.server
import json
import os
import socketserver
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def serve(port):
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=ROOT, **k)
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", port), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("scratch", "wasm_page.png"))
    ap.add_argument("--port", type=int, default=8731)
    ap.add_argument("--limit", default="640")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    httpd = serve(args.port)
    url = "http://127.0.0.1:%d/wasm/" % args.port
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 1400})
        logs = []
        page.on("console", lambda m: logs.append(m.text))
        page.on("pageerror", lambda e: logs.append("PAGEERROR " + str(e)))
        page.goto(url)

        page.wait_for_function("() => document.getElementById('status').textContent.includes('準備完了')",
                               timeout=args.timeout * 1000)
        page.select_option("#limit", args.limit)
        page.click("#btn-sample")
        page.wait_for_function("() => !document.getElementById('btn-run').disabled")
        page.click("#btn-run")
        page.wait_for_function("() => document.getElementById('status').textContent.startsWith('完了')",
                               timeout=args.timeout * 1000)

        # geometry, straight out of the live page: the canvas backing store vs its rendered box,
        # and where the boxes actually are
        geom = page.evaluate("""() => {
          const c = document.getElementById('canvas');
          const r = c.getBoundingClientRect();
          const st = getComputedStyle(c);
          return {backing: [c.width, c.height], rendered: [r.width, r.height],
                  border: st.borderLeftWidth, overlay: !!document.getElementById('overlay')};
        }""")
        info = page.evaluate("""() => {
          const t = document.getElementById('plain').value.split('\\n\\n');
          return {status: document.getElementById('status').textContent, json: t[t.length - 1]};
        }""")
        page.screenshot(path=args.out, full_page=True)
        browser.close()

    httpd.shutdown()
    print("status  :", info["status"])
    print("canvas  : backing %s  rendered %.0fx%.0f  border %s  separate overlay: %s"
          % (geom["backing"], geom["rendered"][0], geom["rendered"][1], geom["border"],
             geom["overlay"]))
    try:
        j = json.loads(info["json"])
        print("lines   :", len(j["lines"]))
        for ln in j["lines"][:6]:
            q = ln["quad"]
            print("   %-22s (%.0f,%.0f)-(%.0f,%.0f)" % (ln["text"], q[0][0], q[0][1], q[2][0], q[2][1]))
    except Exception as e:
        print("could not parse the page JSON:", e)
    for line in logs[:40]:
        print("  console:", line)
    print("screenshot:", args.out)


if __name__ == "__main__":
    main()
