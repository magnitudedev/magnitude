"""Predeclared fixed calibration and two disjoint qualification partitions."""
from program import describe,emit_manifest
from pathlib import Path
import json

def corpus():
    rows=[]
    def add(split,constructor,parameters,**kw):
        name=constructor+'_'+'_'.join(str(v) for v in parameters.values())
        c=describe(name,constructor,parameters,**kw)
        family={'chains':'compute','blocked':'compute','noop':'lifecycle'}.get(constructor,constructor)
        if constructor=='chains' and parameters.get('operation','fma')!='fma':family='integer'
        c.update(family=family,split=split,id=f'{split}-{len(rows):04d}')
        c.update(parameters)
        if constructor in ['sync','strict']:c['op']=parameters['operation']
        if constructor=='atomic':c['op']='cas' if parameters['cas'] else 'uint'
        if constructor=='stream':c['intensity']=parameters.get('intensity',0)
        rows.append(c)
    # Latency, issue, occupancy and liveness: fixed hardware probes.
    for op in ['fma','alu','mul','wide','sfu']:
        for chains in [1,4,16]:
            for threads in [128,2048,8192]:
                add('cal','chains',dict(chains=chains,operation=op),threads=threads,n=threads,it=64)
    for chains in [64,96,112,120,124,128,132,144]:
        add('cal','chains',dict(chains=chains),threads=2048,n=2048,it=64)
    for threads in [128,2048,8192]:
        for shared in [2048,16384,32768]:
            add('cal','chains',dict(chains=32),threads=threads,n=threads,it=64,shared=shared)
    for n in [4096,65536,1048576,8388608]:
        for stride in [1,4]:
            add('cal','stream',dict(intensity=0),n=n,threads=min(n,8192),stride=stride,it=1)
    for n in [1024,16384,262144,4194304]:
        for threads in [128,2048]:add('cal','chase',{},n=n,threads=threads,it=128)
    for op in ['barrier','shuffle','reduce']:
        for threads in [128,2048,8192]:add('cal','sync',dict(operation=op),threads=threads,n=threads,it=128,shared=1024)
    for staged in [False,True]:
        for groups in [4,64,256]:add('cal','matrix',dict(staged=staged),n=groups*128,threads=groups*32,tg=32,it=128,shared=512)
    for cas in [False,True]:
        for n in [1,32,1024,8192]:add('cal','atomic',dict(cas=cas),n=n,threads=8192,it=4)
    for commands in [1,8,32]:add('cal','noop',{},n=128,threads=128,it=1,commands=commands)
    # Helper path structure is extracted from a fixed helper library. Only
    # shared instruction parameters are calibrated; no per-helper coefficient.
    for op in ['add','mul','div','rem','fma','cmp','convert']:
        for mode in [0,2,4,7,8]:add('cal','strict',dict(operation=op),n=16384,threads=2048,it=64,mode=mode)
    # Both partitions are held out. The historical label 'interval' is retained
    # as an identity only; none of its observations calibrate points or bounds.
    for split,ns,it,chs in [('interval',[512,4096],80,[2,8,24,80,126]),('test',[1024,16384],112,[3,12,48,116,136])]:
        for op in ['fma','alu','mul','wide','sfu']:
            for chains in chs[:3]:
                for threads in ns:add(split,'chains',dict(chains=chains,operation=op),threads=threads,n=threads,it=it)
        for threads in ns:
            for chains in chs[3:]:
                for block in [4,16,chains]:
                    add(split,'chains',dict(chains=chains,block=block),threads=threads,n=threads,it=it,equivalence=f'{split}-chains-{threads}-{chains}')
            for shared in [4096,24576]:
                for tg in [64,256]:add(split,'chains',dict(chains=128),threads=threads,n=threads,it=it,shared=shared,tg=tg,equivalence=f'{split}-reservation-{threads}-{shared}')
            for f,u in [(8,8),(24,4),(4,24),(16,16)]:add(split,'mixed',dict(f=f,u=u),threads=threads,n=threads,it=it)
        for n in ([32768,524288,4194304] if split=='interval' else [131072,2097152,16777216]):
            for intensity in [0,8,32]:
                for stride in [1,2]:add(split,'stream',dict(intensity=intensity),n=n,threads=8192,stride=stride,it=1)
        for n in ([8192,131072,2097152] if split=='interval' else [32768,524288,8388608]):
            for threads in ns:add(split,'chase',{},n=n,threads=threads,it=it)
        for op in ['barrier','shuffle','reduce']:
            for threads in ns:
                for tg in [64,256]:add(split,'sync',dict(operation=op),threads=threads,n=threads,it=it,shared=1024,tg=tg)
        for groups in ([16,128] if split=='interval' else [32,512]):
            for staged in [False,True]:add(split,'matrix',dict(staged=staged),n=groups*128,threads=groups*32,tg=32,it=it,shared=512,equivalence=f'{split}-matrix-{groups}')
        for cas in [False,True]:
            for n in ([2,64,512,4096] if split=='interval' else [4,128,2048,8192]):add(split,'atomic',dict(cas=cas),n=n,threads=16384,it=6 if split=='interval' else 8)
        for op in ['add','mul','div','rem','fma','cmp','convert']:
            for mode in [0,1,2,3,4,5,6,7,8]:add(split,'strict',dict(operation=op),n=32768 if split=='interval' else 65536,threads=4096 if split=='interval' else 8192,it=it,mode=mode)
        for commands in [2,16,64]:add(split,'noop',{},n=1024,threads=1024,it=1,commands=commands)
    # Resolve the allocation/residency transition, rather than inferring it
    # from two endpoints. Appending preserves every earlier case identity.
    for chains in [72,80,88,96,112,128]:
        for threads in [128,8192]:
            for tg in [64,256]:
                add('cal','chains',dict(chains=chains),threads=threads,n=threads,tg=tg,it=64,shared=32768 if tg==64 else 16)
    for operation in ['failure','success']:
        for n in [1,32,1024,8192]:add('cal','atomic',dict(cas=False,operation=operation),n=n,threads=8192,it=4)
    for chains in [32,64,96,128]:
        for unroll in [1,2,4,8,16]:
            add('cal','chains',dict(chains=chains,unroll=unroll),threads=2048,n=2048,it=512//unroll,equivalence=f'cal-unroll-{chains}')
    for split,chains_values,iterations in [('interval',[40,88,120],64),('test',[48,80,112],96)]:
        for chains in chains_values:
            for unroll in [3,6,12]:
                add(split,'chains',dict(chains=chains,unroll=unroll),threads=4096,n=4096,it=iterations*6//unroll,equivalence=f'{split}-unroll-{chains}')
    for chains in [16,64]:
        for block in [8,16,32]:
            add('cal','blocked',dict(block=block),chains=chains,mode=chains,threads=2048,n=2048,it=64)
    for split,chains_values,iterations in [('interval',[40,88,120],80),('test',[48,116,136],112)]:
        for chains in chains_values:
            for threads in [1024,8192]:
                for block in [8,16,32]:
                    add(split,'blocked',dict(block=block),chains=chains,mode=chains,threads=threads,n=threads,it=iterations,equivalence=f'{split}-blocked-{chains}-{threads}')
    for chains in [1,4,16]:
        for threads in [128,2048,8192]:add('cal','chains',dict(chains=chains,operation='cmul'),threads=threads,n=threads,it=64)
    for f,u in [(0,16),(16,0)]:
        for threads in [128,2048,8192]:add('cal','mixed',dict(f=f,u=u),threads=threads,n=threads,it=64)
    for operation in [None,'failure','success']:
        for threads in [2048,32768]:
            for n in [1,32,min(8192,threads)]:
                params=dict(cas=False)
                if operation:params['operation']=operation
                add('cal','atomic',params,n=n,threads=threads,it=16)
    for n in [1,32,1024,8192]:add('cal','atomic',dict(cas=True,counted=True),n=n,threads=8192,it=4)
    # Revision 2 qualification is genuinely fresh: none of these configurations
    # were timed for revision 1. All revision-1 observations remain archived.
    widths={2:5,8:10,24:28,80:84,126:130,3:6,12:14,48:56,116:118,136:140,40:44,88:92,120:124,112:108}
    for c in rows:
        if c['split']=='cal':continue
        c['id']='v2-'+c['id'];ctor=c['constructor'];oldthreads=c['threads']
        c['it']=c['it']*3//2
        if ctor in ['chains','blocked','mixed','sync','strict','noop']:
            c['threads']={512:768,4096:6144,1024:1536,16384:12288,8192:10240}[oldthreads]
            c['n']=c['n']*2 if ctor=='strict' else c['threads']
        if ctor=='chains':
            old=c['parameters']['chains'];c['parameters']['chains']=widths.get(old,old)
            if c['parameters'].get('block')==old:c['parameters']['block']=c['parameters']['chains']
            c['chains']=c['parameters']['chains']
        elif ctor=='blocked':c['chains']=widths[c['chains']];c['mode']=c['chains']
        elif ctor=='stream':c['n']=c['n']*3//4;c['threads']=6144;c['stride']=1 if c['stride']==1 else 3
        elif ctor=='chase':c['n']*=2;c['threads']={512:768,4096:6144,1024:1536,16384:12288}[oldthreads]
        elif ctor=='matrix':c['n']=c['n']*3//2;c['threads']=c['threads']*3//2
        elif ctor=='atomic':
            c['threads']=32768;c['n']={2:8,4:16,64:256,128:512,512:1024,2048:4096,4096:8192,8192:16384}[c['n']];c['it']=5 if c['split']=='interval' else 7
        elif ctor=='sync':c['tg']=32 if c['tg']==64 else 128
        elif ctor=='noop':c['commands']={2:3,16:12,64:48}[c['commands']]
        c['kernel']=ctor+'_'+'_'.join(str(v) for v in c['parameters'].values())
        for key,value in c['parameters'].items():c[key]=value
        if c.get('equivalence'):c['equivalence']='v2-'+c['equivalence']
    for operation in ['dependent_failure','dependent_success','dependent_load']:
        for it in [64,256,1024]:add('cal','atomic',dict(cas=False,operation=operation),n=1,threads=1,tg=32,it=it)
    for tg in [32,64,256]:
        for threads in [128,2048,8192]:add('cal','sync',dict(operation='barrier'),threads=threads,n=threads,tg=tg,it=128,shared=1024)
    for c in rows:
        if c['split']=='cal':continue
        c['id']=c['id'].replace('v2-','v3-');ctor=c['constructor']
        c['it']+=48//c['parameters']['unroll'] if 'unroll' in c['parameters'] else 8
        if ctor in ['chains','blocked','mixed','sync','strict','noop','chase']:
            c['threads']={768:1280,6144:5120,1536:1792,12288:14336,10240:9216}[c['threads']]
            if ctor not in ['strict','chase']:c['n']=c['threads']
        elif ctor=='stream':c['n']=c['n']*4//3;c['threads']=5120
        elif ctor=='matrix':c['threads']+=256;c['n']+=1024
        elif ctor=='atomic':c['threads']=49152;c['it']=6 if c['split']=='interval' else 8
        if ctor=='sync' and c['tg']==128:c['tg']=256
        if ctor=='noop':c['commands']={3:4,12:20,48:56}[c['commands']]
        if c.get('equivalence'):c['equivalence']=c['equivalence'].replace('v2-','v3-')
    return rows

if __name__=='__main__':
    rows=corpus();root=Path(__file__).parent/'results/sources';emit_manifest(rows,root)
    emit_manifest([c for c in rows if c['split']=='cal'],root.parent/'calibration-sources')
    emit_manifest([c for c in rows if c['split']!='cal'],root.parent/'qualification-sources')
    print(json.dumps({s:sum(c['split']==s for c in rows) for s in ['cal','interval','test']}))
