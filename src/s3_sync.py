import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from boto3.s3.transfer import TransferConfig


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_prefix(bucket: str, prefix: str, dest_dir: Path, region: Optional[str] = None) -> None:
    s3 = boto3.client("s3", region_name=region)
    paginator = s3.get_paginator("list_objects_v2")
    ensure_dir(dest_dir)

    transfer_cfg = TransferConfig(
        multipart_threshold=8 * 1024 * 1024,
        multipart_chunksize=8 * 1024 * 1024,
        max_concurrency=16,
        use_threads=True,
    )

    found_any = False
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            found_any = True
            rel_key = key[len(prefix) :] if key.startswith(prefix) else key
            dest_path = dest_dir / rel_key
            ensure_dir(dest_path.parent)
            s3.download_file(bucket, key, str(dest_path), Config=transfer_cfg)

    if not found_any:
        # Create empty dir to make downstream steps happy even if no files
        ensure_dir(dest_dir)


def upload_dir(src_dir: Path, bucket: str, dest_prefix: str, region: Optional[str] = None) -> None:
    s3 = boto3.client("s3", region_name=region)
    transfer_cfg = TransferConfig(
        multipart_threshold=8 * 1024 * 1024,
        multipart_chunksize=8 * 1024 * 1024,
        max_concurrency=16,
        use_threads=True,
    )

    if not src_dir.exists():
        return

    for root, _dirs, files in os.walk(src_dir):
        for filename in files:
            file_path = Path(root) / filename
            rel_path = file_path.relative_to(src_dir)
            key = dest_prefix.rstrip("/") + "/" + str(rel_path).replace(os.sep, "/")
            s3.upload_file(str(file_path), bucket, key, Config=transfer_cfg)


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple S3 sync using boto3")
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="Download a prefix to a local directory")
    dl.add_argument("--bucket", required=True)
    dl.add_argument("--prefix", required=True)
    dl.add_argument("--dest", required=True)

    ul = sub.add_parser("upload", help="Upload a local directory to a prefix")
    ul.add_argument("--src", required=True)
    ul.add_argument("--bucket", required=True)
    ul.add_argument("--prefix", required=True)

    args = parser.parse_args()
    region = os.getenv("AWS_REGION")

    try:
        if args.command == "download":
            download_prefix(args.bucket, args.prefix, Path(args.dest), region)
        elif args.command == "upload":
            upload_dir(Path(args.src), args.bucket, args.prefix, region)
        else:
            parser.error("Unknown command")
        return 0
    except (BotoCoreError, ClientError) as e:
        print(f"S3 sync error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


