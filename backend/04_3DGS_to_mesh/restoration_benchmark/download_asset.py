"""Download large public weights in resumable, validated HTTP ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
from pathlib import Path
import re
import shutil
import threading
import time

import requests

from run import PACKAGE_ROOT, dump

ASSETS = {
    'ram': ('models', 'xinyu1205/recognize_anything_model', 'ram_swin_large_14m.pth', 'RAM/ram_swin_large_14m.pth'),
    'seesr_control': ('models', 'cswry/SeeSR', 'seesr/controlnet/diffusion_pytorch_model.safetensors', 'SeeSR/seesr/controlnet/diffusion_pytorch_model.safetensors'),
    'seesr_unet': ('models', 'cswry/SeeSR', 'seesr/unet/diffusion_pytorch_model.safetensors', 'SeeSR/seesr/unet/diffusion_pytorch_model.safetensors'),
}
CHUNK = 8 * 1024 * 1024


def transfer(url, destination, size, expected_sha, workers=3, chunk_size=CHUNK):
    """Publish only a SHA-verified complete file; retain completed ranges on failure."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parts = destination.parent / (destination.name + '.' + expected_sha[:12] + '.parts')
    local = threading.local()
    count = (size + chunk_size - 1) // chunk_size
    state = parts.parent / (parts.name + '.json')

    def length(index):
        return min(chunk_size, size - index * chunk_size)

    def fetch(index):
        target = parts / f'{index:06d}.part'
        if target.exists() and target.stat().st_size == length(index):
            return length(index)
        if not hasattr(local, 'session'):
            local.session = requests.Session()
        start = index * chunk_size
        end = start + length(index) - 1
        for attempt in range(5):
            temporary = target.with_suffix('.tmp')
            try:
                # Resolve a fresh signed CDN URL on every range/retry.
                separator = '&' if '?' in url else '?'
                fresh_url = url + separator + f'download=true&restoration_part={index}&attempt={attempt}'
                with local.session.get(fresh_url, headers={'Range': f'bytes={start}-{end}'},
                                       stream=True, timeout=(15, 60)) as response:
                    response.raise_for_status()
                    wanted = f'bytes {start}-{end}/{size}'
                    if response.status_code != 206 or response.headers.get('Content-Range') != wanted:
                        raise ValueError(f'Invalid range response for part {index}')
                    received = 0
                    with temporary.open('wb') as file:
                        for block in response.iter_content(1024 * 1024):
                            received += len(block)
                            if received > length(index):
                                raise ValueError(f'Excess bytes in part {index}')
                            file.write(block)
                    if received != length(index):
                        raise ValueError(f'Incomplete part {index}: {received}/{length(index)}')
                temporary.replace(target)
                return length(index)
            except (requests.RequestException, ValueError) as error:
                if attempt == 4:
                    message = re.sub(r'https?://[^\s)]+', '[download URL]', str(error))
                    raise RuntimeError(f'Part {index} failed after five attempts: {message}') from None
                time.sleep(min(16, 2 ** attempt))

    with (parts.parent / (parts.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if destination.exists():
            with destination.open('rb') as file:
                digest = hashlib.file_digest(file, 'sha256').hexdigest()
            if digest != expected_sha:
                raise ValueError(f'Existing asset has unexpected SHA256: {destination}')
            return
        parts.mkdir(exist_ok=True)
        completed = 0
        started = time.monotonic()
        last_log = started
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = [pool.submit(fetch, index) for index in range(count)]
            for future in as_completed(futures):
                completed += future.result()
                dump(state, dict(url=url, size=size, sha256=expected_sha,
                                 completed_bytes=completed, expected_parts=count))
                now = time.monotonic()
                if now - last_log >= 30 or completed == size:
                    print(f'{destination.name}: {completed/size:.1%} ({completed/1e9:.2f}/{size/1e9:.2f} GB)', flush=True)
                    last_log = now
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
        temporary = destination.with_suffix(destination.suffix + '.assembling')
        digest = hashlib.sha256()
        with temporary.open('wb') as output:
            for index in range(count):
                with (parts / f'{index:06d}.part').open('rb') as source:
                    for block in iter(lambda: source.read(CHUNK), b''):
                        digest.update(block)
                        output.write(block)
        if temporary.stat().st_size != size or digest.hexdigest() != expected_sha:
            raise ValueError('Assembled asset failed size/SHA256 verification; parts retained')
        temporary.replace(destination)
        dump(state, dict(url=url, size=size, sha256=expected_sha, completed_bytes=size, verified=True))
        shutil.rmtree(parts)
        print(f'Verified asset ready: {destination}; elapsed {time.monotonic()-started:.1f}s', flush=True)


def download_asset(name):
    kind, repo, filename, output = ASSETS[name]
    response = requests.get(f'https://huggingface.co/api/{kind}/{repo}?blobs=true', timeout=30)
    response.raise_for_status()
    info = response.json()
    file = next(f for f in info['siblings'] if f['rfilename'] == filename)
    # Immutable revision plus published SHA prevents mixing checkpoint versions.
    url = f'https://huggingface.co/{repo}/resolve/{info["sha"]}/{filename}'
    transfer(url, PACKAGE_ROOT/'models'/output, file['size'], file['lfs']['sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('asset', choices=ASSETS)
    download_asset(parser.parse_args().asset)


if __name__ == '__main__':
    main()
