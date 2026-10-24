#!/usr/bin/env python3
"""Synthetic 10,384-parameter float32 MLP: 64 -> 128 ReLU -> 16.
Toy weights, not trained; speed/correctness probe, not a language model.
"""
import argparse, json, statistics, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

def weights():
    w1 = np.broadcast_to(((np.arange(64)%4+1)/64).astype(np.float32)[:,None], (64,128)).copy()
    w2 = np.broadcast_to(((np.arange(16)%8-3)/128).astype(np.float32)[None,:], (128,16)).copy()
    return w1, np.zeros(128,dtype=np.float32), w2, np.zeros(16,dtype=np.float32)

def forward(x, w):
    w1,b1,w2,b2 = w
    return np.maximum(x @ w1 + b1, 0) @ w2 + b2

def export(path):
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    w = weights()
    nodes = [helper.make_node('MatMul',['input','w1'],['a']), helper.make_node('Add',['a','b1'],['b']),
             helper.make_node('Relu',['b'],['c']), helper.make_node('MatMul',['c','w2'],['d']), helper.make_node('Add',['d','b2'],['output'])]
    graph = helper.make_graph(nodes,'toy-mlp',[helper.make_tensor_value_info('input',TensorProto.FLOAT,[None,64])],
                              [helper.make_tensor_value_info('output',TensorProto.FLOAT,[None,16])],
                              [numpy_helper.from_array(a,n) for a,n in zip(w,('w1','b1','w2','b2'))])
    model = helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)],producer_name='tiny-nn-probe',ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model,str(path))

def timing(fn, repeats):
    for _ in range(20): fn()
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter_ns(); fn(); samples.append((time.perf_counter_ns()-start)/1e6)
    return {'repetitions': repeats, 'median_ms': statistics.median(samples), 'p95_ms': sorted(samples)[int(.95*(len(samples)-1))],
            'throughput_items_s': 1000/statistics.mean(samples)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--export-onnx'); p.add_argument('--onnx'); p.add_argument('--repeats',type=int,default=1000); a=p.parse_args()
    if a.export_onnx: export(a.export_onnx)
    fixture=json.loads((ROOT/'fixture.json').read_text()); x=np.asarray(fixture['input'],dtype=np.float32); expected=np.asarray(fixture['expected'],dtype=np.float32)
    w=weights(); actual=forward(x,w); np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=0)
    result={'parameters':sum(v.size for v in w),'dtype':'float32','shape':[64,128,16],'batch':1,'expected_ok':True,
            'actual':actual.tolist(),'numpy_version':np.__version__,'numpy':timing(lambda:forward(x,w),a.repeats),
            'method':'20 warmups; independent wall-clock samples; throughput=1000/mean_ms, no batching'}
    if a.onnx:
        import onnxruntime as ort
        options=ort.SessionOptions(); options.intra_op_num_threads=1; options.inter_op_num_threads=1
        session=ort.InferenceSession(a.onnx,sess_options=options,providers=['CPUExecutionProvider'])
        predict=lambda:session.run(None,{'input':x})[0]
        np.testing.assert_allclose(predict(),expected,atol=1e-6,rtol=0)
        result['onnx']={'expected_ok':True,'version':ort.__version__,**timing(predict,a.repeats)}
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
