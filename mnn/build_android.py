#!/usr/bin/env python3
"""Cross-link public MNN 3.6.1 Android CLIs on an x86-64 Linux worker.
Source file hashes and official prebuilt library SHA256 are pinned in inputs.json.
NDK r27d is pinned to Google's published archive checksum.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent


def download(url, destination, expected=None, algorithm='sha256'):
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.new(algorithm)
    with urllib.request.urlopen(url, timeout=120) as response, destination.open('wb') as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
    actual = digest.hexdigest()
    if expected and actual != expected:
        raise RuntimeError(f'{destination.name}: {algorithm} mismatch {actual} != {expected}')
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/mnn')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    work = out / 'work'
    work.mkdir(parents=True, exist_ok=True)
    artifact = out / 'artifact'
    (artifact / 'bin').mkdir(parents=True, exist_ok=True)
    (artifact / 'lib').mkdir(exist_ok=True)
    inputs = json.loads((ROOT / 'inputs.json').read_text())
    source = work / 'source'

    def source_file(entry):
        url = f"https://raw.githubusercontent.com/alibaba/MNN/{inputs['commit']}/{entry['path']}"
        download(url, source / entry['path'], entry['sha256'])
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(source_file, inputs['source_files']))
    download(f"https://raw.githubusercontent.com/alibaba/MNN/{inputs['commit']}/LICENSE",
             artifact / 'MNN-LICENSE.txt')

    libraries = work / 'mnn-official.zip'
    download(inputs['official_android_archive_url'], libraries, inputs['official_archive_sha256'])
    with zipfile.ZipFile(libraries) as archive:
        names = [name for name in archive.namelist() if '/arm64-v8a/' in name and name.endswith('.so')]
        if not names:
            raise RuntimeError('Official archive has no ARM64 shared libraries')
        for name in names:
            with archive.open(name) as incoming, (artifact / 'lib' / Path(name).name).open('wb') as target:
                shutil.copyfileobj(incoming, target)

    ndk_url = 'https://dl.google.com/android/repository/android-ndk-r27d-linux.zip'
    # https://github.com/android/ndk/wiki/Unsupported-Downloads
    algorithm = 'sha1'
    expected = '22105e410cf29afcf163760cc95522b9fb981121'
    ndk_archive = work / 'ndk.zip'
    download(ndk_url, ndk_archive, expected, algorithm)
    # unzip preserves the NDK's executable modes and Unix symlinks.
    subprocess.run(['unzip', '-q', str(ndk_archive), '-d', str(work)], check=True)
    ndk = work / 'android-ndk-r27d'
    compiler = ndk / 'toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++'
    if not compiler.is_file():
        raise RuntimeError('Verified NDK archive did not contain the expected compiler')
    for file in ('NOTICE', 'NOTICE.toolchain'):
        if (ndk / file).is_file():
            shutil.copyfile(ndk / file, artifact / ('NDK-' + file + '.txt'))
    common = [str(compiler), '-std=c++17', '-O3', '-fPIE', '-pie']
    for directory in ('include', 'source', 'tools/cpp', 'transformers/llm/engine/include',
                      'transformers/llm/engine/app', 'transformers/llm/engine/src', '3rd_party'):
        common += ['-I' + str(source / directory)]
    common += ['-L' + str(artifact / 'lib'), '-Wl,-rpath,$ORIGIN/../lib', '-Wl,--allow-shlib-undefined']
    jobs = [
        ('llm_bench', [source / 'transformers/llm/engine/tools/llm_bench.cpp', source / 'tools/cpp/Profiler.cpp']),
        ('llm_demo', [source / 'transformers/llm/engine/demo/llm_demo.cpp']),
        ('edge_mnn', [ROOT / 'edge_mnn.cpp']),
    ]
    commands = []
    for name, files in jobs:
        command = common + list(map(str, files)) + ['-lllm', '-lMNN_Express', '-lMNN', '-llog', '-o', str(artifact / 'bin' / name)]
        commands.append(command)
        subprocess.run(command, check=True)
    hashes = {}
    for path in sorted(artifact.rglob('*')):
        if path.is_file():
            with path.open('rb') as stream:
                hashes[str(path.relative_to(artifact))] = hashlib.file_digest(stream, 'sha256').hexdigest()
    receipt = {'mnn_commit': inputs['commit'], 'mnn_version': inputs['version'],
               'official_library_sha256': inputs['official_archive_sha256'],
               'ndk_version': '27.3.13750724', 'ndk_url': ndk_url,
               'ndk_archive_checksum': {'algorithm': algorithm, 'value': expected},
               'commands': commands, 'artifact_sha256': hashes}
    (out / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
