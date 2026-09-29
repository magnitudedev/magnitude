"""Real Qwen FFN replay observer with generated Metal implementations.

This is an experimental candidate generator, not Seismic code generation.
Timing includes Python/MLX graph construction, submission, and completion.
"""
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

DATA = Path(os.environ.get('QWEN_FEEDBACK_DATA', 'qwen-data'))
ROUND = '''inline float round_bf(float x) {
 uint u=as_type<uint>(x);u+=0x7fff+((u>>16)&1);return as_type<float>(u&0xffff0000);
}'''


def emit(value):
    print(json.dumps(value), flush=True)


class Observer:
    axes = [3, 4, 4, 3, 4, 4, 3]

    def __init__(self, regime):
        if regime not in ['decode', 'prefill32']:
            raise ValueError('unknown regime')
        self.regime = regime
        self.manifest = json.loads((DATA/'manifest.json').read_text())
        self.data = mx.load(str(DATA/'inputs.safetensors'))
        original = mx.load(str(Path(self.manifest['model'])/'model.safetensors'))
        for i in range(self.manifest['layers']):
            for role in ['gate','up','down']:
                for field in ['weight','scales','biases']:
                    self.data[f'l{i}.{role}.{field}'] = original[f'language_model.model.layers.{i}.mlp.{role}_proj.{field}']
        del original
        for i in range(self.manifest['layers']):
            for role in ['gate','up','down']:
                for field in ['scales','biases']:
                    key=f'l{i}.{role}.{field}'
                    self.data[key+'.f32']=self.data[key].astype(mx.float32)
        mx.eval(list(self.data.values()))
        self.reference = dict(np.load(DATA/f'{regime}-reference.npz'))
        self.layers = self.manifest['layers']
        self.m = 1 if regime == 'decode' else 32
        self.artifacts = {}
        self.norm = mx.fast.metal_kernel(name='qwen_feedback_norm', input_names=['x', 'w'],
            output_names=['y'], source='''
            uint lane=thread_position_in_threadgroup.x, row=thread_position_in_grid.y;
            float sum=0;for(uint k=lane;k<2560;k+=32){float v=x[row*2560+k];sum+=v*v;}
            float inv=rsqrt(simd_sum(sum)/2560.0f+1e-6f);
            for(uint k=lane;k<2560;k+=32)y[row*2560+k]=T(x[row*2560+k]*inv*float(w[k]));
            ''')
        self.activation = mx.fast.metal_kernel(name='qwen_feedback_swiglu', input_names=['g', 'u'],
            output_names=['p'], header=ROUND, source='''
            uint i=thread_position_in_grid.x;
            float v=float(g[i]);float a=round_bf(v/(1.0f+exp(-v)));
            p[i]=T(a*float(u[i]));
            ''')
        self.add = mx.fast.metal_kernel(name='qwen_feedback_residual', input_names=['r', 'd'],
            output_names=['y'], source='uint i=thread_position_in_grid.x;y[i]=r[i]+float(d[i]);')

    def kernel(self, n, k, rows, chunk, dual, activated):
        key = (n, k, rows, chunk, dual, activated)
        if key in self.artifacts:
            return self.artifacts[key]
        names = ['x', 'w', 's', 'b'] + (['uw', 'us', 'ub'] if dual else [])
        outputs = ['g', 'u'] if dual else ['g']
        if activated:
            outputs.append('p')
        source = f'''
        uint lane=thread_position_in_threadgroup.x&31;
        uint first=(thread_position_in_grid.x/32)*{rows};
        uint m=thread_position_in_grid.y;
        if(first>={n})return;
        float acc[{rows}]={{0}};
        {'float ua['+str(rows)+']={0};' if dual else ''}
        for(uint base=lane;base<{k//8};base+={32*chunk}){{
          #pragma unroll
          for(uint c=0;c<{chunk};++c){{
            uint word=base+c*32;if(word>={k//8})continue;
            float av[8];
            #pragma unroll
            for(uint j=0;j<8;++j)av[j]=float(x[m*{k}+word*8+j]);
            #pragma unroll
            for(uint r=0;r<{rows};++r){{
              uint row=first+r;
              uint q=w[row*{k//8}+word];uint group=row*{k//64}+word/8;
              float scale=float(s[group]), bias=float(b[group]);
              {'uint uq=uw[row*'+str(k//8)+'+word];float usc=float(us[group]),ubias=float(ub[group]);' if dual else ''}
              #pragma unroll
              for(uint j=0;j<8;++j){{
                float value=fma(float((q>>(4*j))&15),scale,bias);
                acc[r]=fma(av[j],value,acc[r]);
                {'ua[r]=fma(av[j],fma(float((uq>>(4*j))&15),usc,ubias),ua[r]);' if dual else ''}
              }}
            }}
          }}
        }}
        #pragma unroll
        for(uint r=0;r<{rows};++r){{
          float v=round_bf(simd_sum(acc[r]));
          {'float uv=round_bf(simd_sum(ua[r]));' if dual else ''}
          if(lane==0){{
            uint i=m*{n}+first+r;g[i]=T(v);
            {'u[i]=T(uv);' if dual else ''}
            {'p[i]=T(round_bf(v/(1.0f+exp(-v)))*uv);' if activated else ''}
          }}
        }}'''
        name = 'qwen_feedback_' + '_'.join(map(str, key)).replace('True', '1').replace('False', '0')
        fun = mx.fast.metal_kernel(name=name, input_names=names, output_names=outputs, source=source, header=ROUND)
        self.artifacts[key] = fun
        return fun

    def projection(self, prefix, role, x, rows, groups, chunk, dual=False, activated=False):
        n, k = (2560, 9216) if role == 'down' else (9216, 2560)
        fun = self.kernel(n, k, rows, chunk, dual, activated)
        inputs = [x] + [self.data[f'{prefix}.{role}.{f}'] for f in ['weight', 'scales', 'biases']]
        if dual:
            inputs += [self.data[f'{prefix}.up.{f}'] for f in ['weight', 'scales', 'biases']]
        count = 3 if activated else 2 if dual else 1
        grid = math.ceil((n//rows)/groups)*groups*32
        return fun(inputs=inputs, template=[('T', mx.bfloat16)], grid=(grid, self.m, 1),
                   threadgroup=(groups*32, 1, 1), output_shapes=[(self.m,n)]*count,
                   output_dtypes=[mx.bfloat16]*count)

    def graph(self, point, native=False):
        fusion, ri, gi, ci, dri, dgi, dci = point
        rows, groups, chunk = [1,2,4,8][ri], [1,2,4,8][gi], [1,2,4][ci]
        dr, dg, dc = [1,2,4,8][dri], [1,2,4,8][dgi], [1,2,4][dci]
        stages = {}
        previous = None
        for i in range(self.layers):
            pre = f'l{i}'
            residual = self.data[f'{pre}.{self.regime}.residual']
            if previous is not None:
                residual = mx.depends(residual, previous)
            if native:
                normalized = mx.fast.rms_norm(residual, self.data[f'{pre}.norm'].astype(mx.float32), 1e-6).astype(mx.bfloat16)
                def linear(x, role):
                    return mx.quantized_matmul(x.astype(mx.float32), self.data[f'{pre}.{role}.weight'],
                        self.data[f'{pre}.{role}.scales.f32'], self.data[f'{pre}.{role}.biases.f32'],
                        transpose=True, group_size=64, bits=4).astype(mx.bfloat16)
                gate, up = linear(normalized, 'gate'), linear(normalized, 'up')
                # Match the declared publication boundaries explicitly.
                gf = gate.astype(mx.float32)
                act = (gf/(1+mx.exp(-gf))).astype(mx.bfloat16)
                product = (act.astype(mx.float32)*up.astype(mx.float32)).astype(mx.bfloat16)
                down = linear(product, 'down')
                result = residual + down.astype(mx.float32)
            else:
                normalized = self.norm(inputs=[residual, self.data[f'{pre}.norm']], template=[('T',mx.bfloat16)],
                    grid=(32,self.m,1), threadgroup=(32,1,1), output_shapes=[(self.m,2560)], output_dtypes=[mx.bfloat16])[0]
                if fusion:
                    pair = self.projection(pre,'gate',normalized,rows,groups,chunk,True,fusion==2)
                    gate, up = pair[:2]
                else:
                    gate = self.projection(pre,'gate',normalized,rows,groups,chunk)[0]
                    up = self.projection(pre,'up',normalized,rows,groups,chunk)[0]
                if fusion == 2:
                    product = pair[2]
                else:
                    product = self.activation(inputs=[gate,up], template=[('T',mx.bfloat16)],
                        grid=(self.m*9216,1,1), threadgroup=(256,1,1),
                        output_shapes=[(self.m,9216)], output_dtypes=[mx.bfloat16])[0]
                down = self.projection(pre,'down',product,dr,dg,dc)[0]
                result = self.add(inputs=[residual,down], grid=(self.m*2560,1,1),threadgroup=(256,1,1),
                    output_shapes=[(self.m,2560)],output_dtypes=[mx.float32])[0]
            for name,value in [('normalized',normalized),('gate',gate),('up',up),('product',product),('down',down),('result',result)]:
                stages[f'{pre}.{name}'] = value
            previous = result
        return stages

    def execute(self, point, native=False):
        start = time.perf_counter()
        stages = self.graph(point, native)
        mx.eval(list(stages.values()))
        mx.synchronize()
        return (time.perf_counter()-start)*1000, stages

    def check(self, stages):
        max_nrms, max_scaled, checked = 0., 0., 0
        per_stage = {}
        for name, value in stages.items():
            actual = np.asarray(value.astype(mx.float32))
            expected = self.reference[name]
            scale = max(float(np.sqrt(np.mean(expected.astype(np.float64)**2))), 1e-8)
            error = actual.astype(np.float64)-expected
            nrms = float(np.sqrt(np.mean(error**2)))/scale
            scaled = float(np.max(np.abs(error)))/scale
            if not np.isfinite(actual).all() or nrms > .005 or scaled > .05:
                raise RuntimeError(f'{name}: normalized RMS error {nrms}, max/RMS {scaled}; limits .005/.05')
            key = name.split('.')[-1]
            per_stage[key] = max(per_stage.get(key,0.), nrms)
            max_nrms, max_scaled = max(max_nrms,nrms), max(max_scaled,scaled)
            checked += expected.size
        return dict(max_normalized_rms_error=max_nrms,max_error=max_scaled,checked_elements=checked,stage_errors=per_stage)

    def observe(self, point, trials=3, target_ms=2., native=False):
        if len(point)!=len(self.axes) or any(not 0<=x<n for x,n in zip(point,self.axes)):
            raise ValueError('coordinate outside domain')
        start = time.perf_counter()
        old = len(self.artifacts)
        # The first complete invocation includes lazy JIT and first execution.
        first, stages = self.execute(point,native)
        new = len(self.artifacts)-old
        samples = [self.execute(point,native)[0] for _ in range(trials)]
        checks = self.check(stages)
        first_cost = first if new else 0.
        return dict(point=point,samples_ms=samples,median_ms=statistics.median(samples),
            compile_ms=first_cost,compile_metric='first complete invocation when new custom artifacts exist; includes execution',
            new_artifacts=new,artifact_count=len(self.artifacts),first_invocation_ms=first,
            wall_ms=(time.perf_counter()-start)*1000,repetitions=1,**checks)


def main():
    started = time.perf_counter()
    observer = Observer(sys.argv[1])
    emit(dict(kind='ready',fixture=sys.argv[1],axes=observer.axes,layers=observer.layers,
              initialization_ms=(time.perf_counter()-started)*1000,
              endpoint='wall duration: construct, submit, complete all 32 captured FFN invocations',
              layer_order='explicit dependency from each prior FFN result to the next captured input',
              numerical_contract='BF16 intermediate publication, reassociated F32 dots; validation NRMSE<=.005 and max/RMS<=.05 per stage',
              model_revision=observer.manifest['revision']))
    for line in sys.stdin:
        try:
            request = json.loads(line)
            emit(observer.observe(request['point'],request.get('trials',3),request.get('target_ms',2),request.get('native',False)))
        except Exception as e:
            emit(dict(error=str(e)))


if __name__ == '__main__':
    main()
