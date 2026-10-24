#!/usr/bin/env python3
"""Read-only, bounded hardware probe. No environment variables or private files."""
import importlib.metadata as md
import json, os, platform, resource, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

def text(path):
    try: return Path(path).read_text().strip()
    except OSError as e: return {'unavailable': str(e)}

def command(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        return {'returncode': p.returncode, 'stdout': p.stdout[-20000:], 'stderr': p.stderr[-2000:]}
    except Exception as e: return {'unavailable': str(e)}

def probe():
    packages = {}
    for name in ('numpy', 'onnx', 'onnxruntime', 'torch', 'transformers', 'llama-cpp-python'):
        try: packages[name] = md.version(name)
        except md.PackageNotFoundError: packages[name] = None
    accelerators = {'nvidia_smi': command(['nvidia-smi']), 'lspci': command(['lspci']),
                    'opencl': command(['clinfo']), 'opencl_libraries': command(['ldconfig', '-p']),
                    'vulkan': command(['vulkaninfo', '--summary']), 'torch_installed': packages['torch'] is not None}
    if packages['torch']:
        accelerators['torch_runtime'] = command([sys.executable, '-c', 'import json,torch; print(json.dumps(dict(version=torch.__version__,cuda_available=torch.cuda.is_available(),cuda_version=torch.version.cuda,devices=torch.cuda.device_count(),mps_available=bool(hasattr(torch.backends,"mps") and torch.backends.mps.is_available()))))'])
    egress = {}
    for url in ('https://huggingface.co/api/models/bartowski/SmolLM2-135M-Instruct-GGUF', 'https://api.github.com/repos/kvnloo/tiny-nn-probe'):
        start = time.monotonic()
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                data = r.read(1048576)
                egress[url] = {'status': r.status, 'bytes': len(data), 'seconds': time.monotonic()-start}
        except Exception as e: egress[url] = {'error': str(e), 'seconds': time.monotonic()-start}
    disk = shutil.disk_usage('.')
    limits = {name: resource.getrlimit(getattr(resource,name)) for name in ('RLIMIT_CPU','RLIMIT_AS','RLIMIT_FSIZE')}
    return {'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), 'os': platform.platform(),
            'uname': list(platform.uname()), 'python': sys.version, 'logical_cores': os.cpu_count(),
            'affinity': sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            'cpu': text('/proc/cpuinfo'), 'memory': text('/proc/meminfo'),
            'cgroup_memory_max': text('/sys/fs/cgroup/memory.max'), 'disk_bytes': dict(zip(('total','used','free'), disk)),
            'packages': packages, 'accelerators': accelerators, 'network_egress': egress, 'resource_limits': limits,
            'wall_time_limit': 'Unknown: no timeout stress test; RLIMIT_CPU is not a sandbox wall limit.'}

if __name__ == '__main__': print(json.dumps(probe(), indent=2))
