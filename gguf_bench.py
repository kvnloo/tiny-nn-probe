#!/usr/bin/env python3
"""Pinned GGUF ladder; llama-bench throughput + unconstrained classification.
Run only on an explicitly admitted worker. No implicit compilation or GPU selection.
"""
import argparse, hashlib, json, os, re, socket, subprocess, time, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ITEMS=[
 ('Which option is an animal? A: blue B: cat C: France','B'),
 ('Which option is a color? A: green B: dog C: Japan','A'),
 ('Which option is a country? A: horse B: red C: Canada','C'),
 ('Classify: "Thank you for your help!" A: warning B: gratitude C: farewell','B'),
 ('Classify: "Goodbye!" A: farewell B: greeting C: apology','A'),
 ('Classify: "Error: file not found." A: greeting B: gratitude C: error report','C'),
 ('Which option equals "blue"? A: green B: blue C: red','B'),
 ('Classify: "Hello there!" A: greeting B: command C: error report','A'),
]

def save(path,data): path.write_text(json.dumps(data,indent=2)+'\n')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def download(model,cache,max_bytes):
    target=cache/model['file']
    url=f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{model['file']}"
    if target.exists():
        if max_bytes and target.stat().st_size>max_bytes: raise MemoryError('cached model exceeds configured model-file budget')
        return target,url
    part=target.with_suffix('.part')
    try:
        with urllib.request.urlopen(url,timeout=120) as r, part.open('wb') as f:
            length=r.headers.get('Content-Length')
            if length and max_bytes and int(length)>max_bytes: raise MemoryError(f'model file {length} bytes exceeds model-file budget {max_bytes}; runtime overhead additional')
            size=0
            while True:
                chunk=r.read(1024*1024)
                if not chunk: break
                size+=len(chunk)
                if max_bytes and size>max_bytes: raise MemoryError('download exceeds model-file budget; runtime overhead additional')
                f.write(chunk)
        part.replace(target)
    finally:
        part.unlink(missing_ok=True)
    return target,url

def post(port,payload):
    req=urllib.request.Request(f'http://127.0.0.1:{port}/completion',json.dumps(payload).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180) as r: return json.load(r)

def classify(server,model,threads,ngl,out,timeout):
    with socket.socket() as s: s.bind(('127.0.0.1',0)); port=s.getsockname()[1]
    cmd=[str(server),'-m',str(model),'-t',str(threads),'-ngl',str(ngl),'-c','512','--host','127.0.0.1','--port',str(port),'-np','1']
    rows=[]
    with (out/'server.log').open('w') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+timeout
            while time.monotonic()<deadline:
                if proc.poll() is not None: raise RuntimeError(f'server exited {proc.returncode}')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1) as r:
                        if r.status==200: break
                except Exception: time.sleep(.25)
            else: raise TimeoutError('server startup timeout')
            for question,expected in ITEMS:
                prompt='<|im_start|>system\nChoose the answer. Reply with only A, B, or C. No explanation.<|im_end|>\n<|im_start|>user\n'+question+'<|im_end|>\n<|im_start|>assistant\n'
                if 'Qwen3' in model.name: prompt+='<think>\n\n</think>\n\n'
                start=time.monotonic()
                response=post(port,{'prompt':prompt,'n_predict':8,'temperature':0,'seed':1729,'cache_prompt':False,'stop':['<|im_end|>','\n']})
                content=response.get('content','').strip(); match=re.match(r'^([ABC])(?:\b|[.:)])',content)
                actual=match.group(1) if match else None
                rows.append({'question':question,'expected':expected,'content':content,'actual':actual,'correct':actual==expected,'wall_seconds':time.monotonic()-start,'timings':response.get('timings'), 'response':response})
                save(out/'classification.json',{'rows':rows,'correct':sum(x['correct'] for x in rows),'total':len(rows),'grammar_constrained':False})
        finally:
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
    return {'correct':sum(x['correct'] for x in rows),'total':len(rows),'grammar_constrained':False}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--bin-dir',required=True); p.add_argument('--out',default='results/gguf'); p.add_argument('--cache',default='models'); p.add_argument('--threads',type=int,default=2); p.add_argument('--gpu-layers',type=int,default=0); p.add_argument('--max-model-mib',type=int,default=0); p.add_argument('--models',default=''); p.add_argument('--timeout',type=int,default=240); a=p.parse_args()
    out=Path(a.out).resolve(); out.mkdir(parents=True,exist_ok=True); cache=Path(a.cache).resolve(); cache.mkdir(parents=True,exist_ok=True)
    bindir=Path(a.bin_dir).resolve(); bench=bindir/'llama-bench'; server=bindir/'llama-server'; results=[]
    models=json.loads((ROOT/'models.json').read_text())
    for model in models:
        if a.models and model['id'] not in a.models.split(','): continue
        folder=out/model['id']; folder.mkdir(exist_ok=True)
        row={**model,'threads':a.threads,'gpu_layers':a.gpu_layers,'affinity':sorted(os.sched_getaffinity(0)),'status':'starting'}; start=time.monotonic()
        try:
            path,url=download(model,cache,a.max_model_mib*1024*1024); row.update(url=url,bytes=path.stat().st_size,sha256=sha(path))
            cmd=[str(bench),'-m',str(path),'-p','128','-n','32','-r','3','-t',str(a.threads),'-ngl',str(a.gpu_layers),'-o','json']
            row['benchmark_command']=cmd
            run=subprocess.run(cmd,capture_output=True,text=True,timeout=a.timeout)
            (folder/'bench.stdout').write_text(run.stdout); (folder/'bench.stderr').write_text(run.stderr)
            if run.returncode: raise RuntimeError(f'llama-bench exit {run.returncode}')
            data=json.loads(run.stdout); save(folder/'bench.json',data)
            pp=[r for r in data if r.get('n_prompt',0)>0 and r.get('n_gen',0)==0]; tg=[r for r in data if r.get('n_gen',0)>0 and r.get('n_prompt',0)==0]
            if not pp or not tg: raise ValueError('missing separate prompt/decode benchmark rows')
            row.update(pp_tok_s=pp[0]['avg_ts'],decode_tok_s=tg[0]['avg_ts'],classification=classify(server,path,a.threads,a.gpu_layers,folder,a.timeout),status='measured')
        except Exception as e:
            row.update(status='budget_stop' if isinstance(e,MemoryError) else 'failed',error=f'{type(e).__name__}: {e}')
        row['wall_seconds']=time.monotonic()-start; results.append(row); save(out/'summary.json',results); print(json.dumps(row),flush=True)
        if row['status']!='measured' or row['decode_tok_s']<5: break

if __name__=='__main__': main()
