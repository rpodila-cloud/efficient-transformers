"""Logical-history fixtures shared across layout choices; CPU and optional device checks."""
from dataclasses import asdict
from contextlib import contextmanager
from pathlib import Path
import json


def config(w, c):
    from ..forwards import Config
    keys={k:v for k,v in asdict(w).items() if k in Config.__dataclass_fields__}
    return Config(profile='m3',impl='msa_indexer_attention_GP_unified',**keys,index_local_blocks=1,
                  num_kv_blocks=c.chunks,msa_indexer_dp=c.msa_indexer_dp,msa_indexer_cp=c.msa_indexer_cp,
                  msa_attn_dp=c.msa_attn_dp,msa_attn_cp=c.msa_attn_cp,num_cores_per_device=c.num_cores,skip_kv=False)


def pack_history(logical, dp, cp, unit):
    """[B,H,T,D] chronological -> [B/DP,DP*CP*H,T/CP,D]."""
    import torch
    b,h,t,d=logical.shape
    if b%dp or t%(cp*unit):
        raise ValueError('history does not divide the requested ownership layout')
    out=torch.empty((b//dp,dp*cp*h,t//cp,d),dtype=logical.dtype)
    owners=(torch.arange(t)//unit)%cp
    for lane in range(dp):
        for way in range(cp):
            out[:,(lane*cp+way)*h:(lane*cp+way+1)*h]=logical[lane*(b//dp):(lane+1)*(b//dp),:,owners==way,:]
    return out


def unpack_history(packed,batch,heads,dp,cp,unit):
    import torch
    local,rows,slots,dim=packed.shape
    out=torch.empty((batch,heads,slots*cp,dim),dtype=packed.dtype)
    owners=(torch.arange(slots*cp)//unit)%cp
    for lane in range(dp):
        for way in range(cp):
            out[lane*local:(lane+1)*local,:,owners==way,:]=packed[:,(lane*cp+way)*heads:(lane*cp+way+1)*heads]
    return out


def tiny_check(w, c, positions=(127,128,129), export_dir=None):
    """Explicit CPU validation workload. No production latency/precision claims."""
    import torch
    from .. import bench
    from ..parity import Reference
    from layer_benchmarks.common import custom_ops, parity
    torch.set_num_threads(4)
    from .accuracy import metrics, tensor_manifest, FP32_CPU_CRITERION, FP32_EXPORT_CRITERION, FP32_CACHE_CRITERION, selection_properties
    cfg=config(w,c)
    model=bench.build(cfg,0,1,torch.float32)
    ref=Reference(cfg,model,w.batch_size)
    gen=torch.Generator().manual_seed(932)
    logical=[torch.randn(x.shape,generator=gen)*0.05 for x in ref.msa[0]]
    ref.msa[0]=tuple(x.clone() for x in logical)
    feeds=bench.inputs(cfg,w.batch_size,1,positions[0],torch.float32,0,1)
    states=bench.cache_names(0,1); names,outnames=bench.io_names(cfg,0,1,1)
    for name,value in zip(states,logical):
        index=name.startswith('index')
        feeds[name]=pack_history(value,c.msa_indexer_dp if index else c.msa_attn_dp,
                                 c.msa_indexer_cp if index else c.msa_attn_cp,128 if index else 1)
    checks=[]; export_checks={}
    initial_manifest={'weights':tensor_manifest(model.state_dict()),
                      'logical_history':tensor_manifest(dict(zip(states,logical))),
                      'physical_inputs':tensor_manifest(feeds)}
    for n,v in zip(states,logical):
        index=n.startswith('index')
        verify_pack_coordinates(v,feeds[n],c.msa_indexer_dp if index else c.msa_attn_dp,
                                c.msa_indexer_cp if index else c.msa_attn_cp,128 if index else 1,
                                sorted(set(positions)|{0,1,127,128,255,256,w.ctx_len-1}))
    with torch.no_grad(),custom_ops.enabled(False):
        for pos in positions:
            feeds['position_ids']=torch.full((w.batch_size,1),pos,dtype=torch.int32)
            feeds['hidden_states']=torch.randn(w.batch_size,1,w.hidden_size,generator=gen)
            # Unified non-paged decode ABI includes external RoPE buffers.
            if 'msa_cos' in feeds:
                feeds['msa_cos'],feeds['msa_sin']=bench._prefill_rope(cfg,feeds['position_ids'],torch.float32)
            before={k:v.clone() for k,v in feeds.items()}
            with capture_unified_selection() as selection:
                out=model(*(feeds[n] for n in names))
            want=ref(feeds['hidden_states'],feeds['position_ids'])
            torch.testing.assert_close(out[0],want,atol=1e-4,rtol=1e-3)
            step={'position':pos,'max_abs':float((out[0]-want).abs().max()),
                  'output':metrics(out[0],want,FP32_CPU_CRITERION),
                  'input_manifest':tensor_manifest(before),'selection':[], 'caches':{}}
            checks.append(step)
            for ids,valid in zip(selection['ids'].reshape(-1,cfg.selected_blocks),selection['valid'].reshape(-1,cfg.selected_blocks)):
                checked=selection_properties(ids,pos,w.ctx_len,valid)
                if not checked['passed']:raise AssertionError(checked)
                step['selection'].append(checked)
            ids=model.msa_layers[0].last_block_indices
            if not torch.all(ids[...,-1]==pos//128):
                raise AssertionError('current block is missing')
            for i,(name,value) in enumerate(zip(states,out[1:])):
                index=name.startswith('index')
                logical_out=unpack_history(value,w.batch_size,w.indexer_n_head if index else w.num_kv_heads,
                                            c.msa_indexer_dp if index else c.msa_attn_dp,
                                            c.msa_indexer_cp if index else c.msa_attn_cp,128 if index else 1)
                torch.testing.assert_close(logical_out,ref.msa[0][i],atol=1e-5,rtol=1e-4)
                prev=unpack_history(before[name],w.batch_size,w.indexer_n_head if index else w.num_kv_heads,
                                    c.msa_indexer_dp if index else c.msa_attn_dp,
                                    c.msa_indexer_cp if index else c.msa_attn_cp,128 if index else 1)
                unchanged=torch.cat((logical_out[:,:,:pos],logical_out[:,:,pos+1:]),dim=2)
                prior=torch.cat((prev[:,:,:pos],prev[:,:,pos+1:]),dim=2)
                if not torch.equal(unchanged,prior):raise AssertionError('non-current cache contents changed')
                step['caches'][name]={'vs_reference':metrics(logical_out,ref.msa[0][i],FP32_CACHE_CRITERION),
                                      'all_noncurrent_exact':True}
                feeds[name]=value
        if export_dir is not None:
            export_dir=Path(export_dir); export_dir.mkdir(parents=True,exist_ok=True)
            path=bench.export_onnx(model,before,(names,outnames),export_dir/'reference.onnx',custom_ops=False,optimize=w.onnx_optimize)
            got=parity.ort_run(path,{n:before[n].numpy() for n in names},outnames)
            for name,value in zip(outnames,out):
                torch.testing.assert_close(torch.from_numpy(got[name]),value,atol=2e-5,rtol=1e-4)
                export_checks[name]=metrics(got[name],value,FP32_EXPORT_CRITERION)
    return {'status':'passed','scope':'FP32 tiny independent reference; nonzero retained history, all cache outputs, current block and ownership boundaries',
            'effective':c.effective(),'workload':asdict(w),'checks':checks,'onnx_parity':export_dir is not None,
            'initial_manifest':initial_manifest,'export_checks':export_checks,
            'export_scope':'CPU decomposed graph, optimized='+str(w.onnx_optimize)+'; not actual custom-op production ONNX',
            'hardware_accuracy_status':'not_established'}


def verify_pack_coordinates(logical,packed,dp,cp,unit,positions):
    import torch
    b,h,t,d=logical.shape;local=b//dp
    for batch in range(b):
        for head in range(h):
            for p in positions:
                if p<0 or p>=t:continue
                row=((batch//local)*cp+(p//unit)%cp)*h+head
                slot=(p//(unit*cp))*unit+p%unit
                if not torch.equal(logical[batch,head,p],packed[batch%local,row,slot]):
                    raise ValueError(f'cache mapping mismatch at batch{batch}, head{head}, token{p}, row{row}, slot{slot}')



@contextmanager
def capture_unified_selection():
    """Read actual eager selection validity without changing production math."""
    import sys
    previous=sys.gettrace(); result={}
    def trace(frame,event,arg):
        if event=='return' and frame.f_code.co_name=='msa_indexer_attention_GP_unified' and frame.f_globals.get('__name__','').endswith('msa.forwards'):
            result['ids']=frame.f_locals['block_indices'].detach().clone()
            result['valid']=frame.f_locals['token_valid'].detach().clone()
        return trace
    sys.settrace(trace)
    try:yield result
    finally:sys.settrace(previous)
