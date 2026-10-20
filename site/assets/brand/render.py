"""Rebuild this project's poster, animated GIF and H.264 MP4.

Install: pip install playwright imageio-ffmpeg
         playwright install chromium
Run:     python site/assets/brand/render.py
"""
import argparse
import base64
import functools
import http.server
from pathlib import Path
import subprocess
import tempfile
import threading

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="Export only cover.png.")
    parser.add_argument("--browser", help="Optional installed Chromium/Edge executable.")
    args = parser.parse_args()
    brand = Path(__file__).resolve().parent
    site = brand.parents[1]

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Handler, directory=str(site)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(**({"executable_path":args.browser} if args.browser else {}))
            page = browser.new_page(viewport={"width":1600,"height":820})
            page.goto(f"http://127.0.0.1:{server.server_port}/assets/brand/render.html")
            page.evaluate("window.posterReady")
            def frame(t):
                return base64.b64decode(page.evaluate(
                    "(t)=>{renderPoster(t);return document.querySelector('canvas').toDataURL('image/png').split(',')[1]}",t))
            (brand/"cover.png").write_bytes(frame(11.2))
            if not args.preview:
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
                with tempfile.TemporaryDirectory(prefix="research-cover-") as tmp:
                    frames = Path(tmp)
                    for n in range(144):
                        (frames/f"{n:04d}.png").write_bytes(frame(n/12))
                    common=[ffmpeg,"-hide_banner","-loglevel","error","-y",
                            "-framerate","12","-i",str(frames/"%04d.png")]
                    subprocess.run(common+["-c:v","libx264","-crf","19","-preset","medium",
                        "-pix_fmt","yuv420p","-movflags","+faststart","-an",str(brand/"cover.mp4")],check=True)
                    subprocess.run(common+["-vf",
                        "scale=1200:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                        "-loop","0",str(brand/"cover.gif")],check=True)
            browser.close()
    finally:
        server.shutdown()
    print("Exported cover assets to",brand)

if __name__ == "__main__":
    main()
