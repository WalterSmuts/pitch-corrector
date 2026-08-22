#!/usr/bin/env python3
"""Static dev server that sets the cross-origin isolation headers required by
the AudioWorklet (`worklet` feature) build.

The AudioWorklet host needs `crossOriginIsolated === true` (for SharedArrayBuffer
+ wasm threads), which requires these response headers:

    Cross-Origin-Opener-Policy: same-origin
    Cross-Origin-Embedder-Policy: require-corp

Python's default ``http.server`` does not send them, so the worklet host's
``is_available()`` returns false. Use this instead:

    python3 serve.py 8888

The default ScriptProcessorNode build does not need this server.
"""
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class COOPCOEPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        # Allow the wasm/js assets to be embedded under COEP.
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        # Without this, browsers apply heuristic freshness and may serve
        # some ES modules from cache while fetching others fresh — mixed
        # file versions after an edit ("X is not a function" errors).
        # no-cache still allows 304 revalidation, so reloads stay fast.
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8888
    server = ThreadingHTTPServer(("127.0.0.1", port), COOPCOEPRequestHandler)
    print(f"Serving with COOP/COEP on http://localhost:{port} (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
