"""Independent controls: ordinary BF16 MLX and transfer to real prefill inputs."""
import argparse
import json
import statistics
from pathlib import Path

import mlx.core as mx
from qwen_worker import Observer


def mlx_bf16(observer):
    stages={}
    previous=None
    for i in range(observer.layers):
        pre=f'l{i}'
        residual=observer.data[f'{pre}.{observer.regime}.residual']
        if previous is not None:residual=mx.depends(residual,previous)
        normalized=mx.fast.rms_norm(residual,observer.data[f'{pre}.norm'].astype(mx.float32),1e-6).astype(mx.bfloat16)
        def linear(x,role):
            return mx.quantized_matmul(x,observer.data[f'{pre}.{role}.weight'],
                observer.data[f'{pre}.{role}.scales'],observer.data[f'{pre}.{role}.biases'],
                transpose=True,group_size=64,bits=4)
        gate,up=linear(normalized,'gate'),linear(normalized,'up')
        gf=gate.astype(mx.float32)
        product=((gf/(1+mx.exp(-gf))).astype(mx.bfloat16).astype(mx.float32)*up.astype(mx.float32)).astype(mx.bfloat16)
        down=linear(product,'down');result=residual+down.astype(mx.float32)
        for name,value in [('normalized',normalized),('gate',gate),('up',up),('product',product),('down',down),('result',result)]:
            stages[f'{pre}.{name}']=value
        previous=result
    return stages


def main():
    import time
    p=argparse.ArgumentParser()
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    selection=json.loads(a.summary.read_text())['best_confirmed_point']
    reports=[]
    for regime in ['decode','prefill32']:
        observer=Observer(regime)
        for mode in ['custom_decode_selection','mlx_matched_f32','mlx_bf16']:
            values=[]
            for repeat in range(7):
                if mode=='mlx_bf16':
                    start=time.perf_counter();stages=mlx_bf16(observer)
                    mx.eval(list(stages.values()));mx.synchronize()
                    elapsed=(time.perf_counter()-start)*1000
                else:
                    elapsed,stages=observer.execute(selection,native=mode=='mlx_matched_f32')
                if repeat:values.append(elapsed)
            try:
                checks=observer.check(stages);accepted=True
            except RuntimeError as e:
                checks={'rejection':str(e)};accepted=False
            report=dict(regime=regime,mode=mode,samples_ms=values,mean_ms=statistics.mean(values),
                        accepted_by_experiment_contract=accepted,**checks)
            reports.append(report);print(json.dumps(report),flush=True)
    a.output.write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':main()
