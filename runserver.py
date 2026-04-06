"""
Start the FIG-Loneliness FastAPI server.

Usage
─────
  # Development (auto-reload)
  python serve.py

  # Production
  python serve.py --host 0.0.0.0 --port 8000 --workers 2
"""

import argparse
import uvicorn


def parse_args():
    parser = argparse.ArgumentParser(description="Start the FIG-Loneliness API")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Bind port (default: 8000)"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload (dev mode)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level="info",
    )
