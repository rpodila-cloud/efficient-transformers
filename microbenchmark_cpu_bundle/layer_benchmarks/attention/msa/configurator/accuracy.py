"""Content provenance and diagnostic metrics; no implicit hardware acceptance policy."""
from __future__ import annotations
import hashlib
import numpy as np

FP32_CPU_CRITERION = {
    'name': 'existing_fp32_eager_reference', 'atol': 1e-4, 'rtol': 1e-3,
    'provenance': 'msa/parity.py:_case constructs torch.float32 model and inputs',
    'scope': 'FP32 CPU algorithm/reference parity; provisional diagnostic only for other precisions',
}
FP32_EXPORT_CRITERION = {
    'name': 'existing_fp32_eager_ort', 'atol': 2e-5, 'rtol': 1e-4,
    'provenance': 'msa/parity.py:_case decomposed export, FP32 CPU',
    'scope': 'FP32 CPU export parity; not an FP16/MXFP6 hardware policy',
}
FP32_CACHE_CRITERION = {
    'name': 'existing_configurator_fp32_cache', 'atol': 1e-5, 'rtol': 1e-4,
    'provenance': 'configurator/validation.py:tiny_check, phase-5 CPU regression assertion; original derivation not recorded',
    'scope': 'FP32 CPU updated-cache comparison only; unchanged cache contents require exact equality; not a hardware policy',
}


def numpy_value(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
        if str(value.dtype) == 'torch.bfloat16':
            value = value.float()  # metric conversion only; fingerprint preserves raw dtype below
        return value.numpy()
    return np.asarray(value)


def fingerprint(value):
    """Hash exact stored representation, shape and dtype separately (no seed inference)."""
    if hasattr(value, 'detach'):
        import torch
        value = value.detach().cpu().contiguous()
        dtype, shape = str(value.dtype).removeprefix('torch.'), list(value.shape)
        raw = value.reshape(-1).view(torch.uint8).numpy()
    else:
        value = np.asarray(value)
        dtype, shape = str(value.dtype), list(value.shape)
        value = np.ascontiguousarray(value)
        raw = value.reshape(-1).view(np.uint8)
    h = hashlib.sha256()
    view = memoryview(raw)
    for off in range(0, len(view), 8*1024**2):
        h.update(view[off:off+8*1024**2])
    return {'dtype': dtype, 'shape': shape, 'bytes': raw.size, 'sha256': h.hexdigest()}


def tensor_manifest(values):
    return {name: fingerprint(value) for name, value in sorted(values.items())}


def require_same_contents(actual, expected):
    if set(actual) != set(expected):
        raise ValueError('tensor names differ')
    different = [name for name in actual if actual[name] != expected[name]]
    if different:
        raise ValueError('tensor content/dtype/shape differs: '+', '.join(different))


def _validate_buffer_contract(arrays, bindings, direction):
    if set(arrays) != set(bindings):
        raise ValueError(f'runtime {direction} names differ from captured arrays')
    for name, value in arrays.items():
        value = np.asarray(value); binding = bindings[name]
        if (list(value.shape) != list(binding['shape']) or
                str(value.dtype) != str(np.dtype(binding['dtype'])) or
                value.nbytes != binding['bytes']):
            raise ValueError(f'runtime {direction} shape/dtype/bytes mismatch: '+name)
    return {'passed': True, direction+'_names': sorted(arrays)}


def validate_submission_contract(arrays, bindings):
    """Reject ignored/missing inputs and representation mismatches before run()."""
    return _validate_buffer_contract(arrays, bindings, 'input')


def validate_output_contract(arrays, bindings):
    """Missing retained-state payloads cannot become a successful capture."""
    return _validate_buffer_contract(arrays, bindings, 'output')


def cache_state_check(actual, before, positions, dp, cp, unit, expected_updates):
    """Exact non-current byte checks; current-entry metrics are provisional."""
    actual, before = np.asarray(actual), np.asarray(before)
    if (actual.shape != before.shape or actual.dtype != before.dtype or before.ndim != 4 or
            any(type(v) is not int or v <= 0 for v in (dp,cp,unit)) or
            min(before.shape) <= 0 or before.shape[1] % (dp*cp) or before.shape[2] % unit):
        raise ValueError('cache state shape/dtype/ownership mismatch')
    local, rows, slots, dim = before.shape
    heads = rows//(dp*cp)
    positions = np.asarray(positions)
    if (positions.shape != (local*dp, 1) or not np.issubdtype(positions.dtype, np.integer) or
            (positions < 0).any() or (positions >= slots*cp).any()):
        raise ValueError('cache update positions do not match the cache')
    updates, allowed = [], {}
    for batch, pos in enumerate(positions[:, 0]):
        owner, slot = (int(pos)//unit)%cp, (int(pos)//(unit*cp))*unit+int(pos)%unit
        if slot >= slots:
            raise ValueError('cache local update slot out of range')
        first = ((batch//local)*cp+owner)*heads
        updates.append(actual[batch%local, first:first+heads, slot])
        for row in range(first, first+heads):
            allowed[batch%local, row] = slot
    if np.asarray(expected_updates).shape != (local*dp, heads, dim):
        raise ValueError('reference current-update shape mismatch')
    changed, nonfinite = 0, 0
    chunk_slots = max(1, (8*1024**2)//(dim*before.dtype.itemsize))
    for batch in range(local):
        for row in range(rows):
            update_slot = allowed.get((batch,row))
            for start in range(0,slots,chunk_slots):
                end = min(start+chunk_slots,slots)
                a = np.ascontiguousarray(actual[batch,row,start:end])
                b = np.ascontiguousarray(before[batch,row,start:end])
                nonfinite += int(np.count_nonzero(~np.isfinite(a)))
                a = a.view(np.uint8).reshape(end-start,-1)
                b = b.view(np.uint8).reshape(end-start,-1)
                if update_slot is not None and start <= update_slot < end:
                    offset = update_slot-start
                    changed += int(np.count_nonzero(a[:offset] != b[:offset]))
                    changed += int(np.count_nonzero(a[offset+1:] != b[offset+1:]))
                else:
                    changed += int(np.count_nonzero(a != b))
    return {'passed': changed == 0 and nonfinite == 0,
            'passed_scope': 'finiteness and exact non-current bytes only; current-entry numerics are separate provisional diagnostics',
            'noncurrent_changed_bytes': changed, 'nonfinite_elements': nonfinite,
            'noncurrent_checked_bytes': before.nbytes-len(allowed)*dim*before.dtype.itemsize,
            'method': 'exact bytes outside DP/CP current-token coordinates; includes signed-zero changes',
            'current_update_check': metrics(np.stack(updates),expected_updates,FP32_CACHE_CRITERION),
            'hardware_accuracy_status': 'unverified'}


def metrics(actual, expected, criterion=None):
    """Float64 error accumulation; criterion results do not imply hardware accuracy."""
    if criterion is not None:
        atol, rtol = criterion['atol'], criterion['rtol']
        if not np.isfinite(atol) or not np.isfinite(rtol) or min(atol, rtol) < 0:
            raise ValueError('invalid criterion')
    a, e = numpy_value(actual), numpy_value(expected)
    result = {'actual_shape': list(a.shape), 'expected_shape': list(e.shape),
              'actual_dtype': str(a.dtype), 'expected_dtype': str(e.dtype),
              'shape_equal': a.shape == e.shape, 'criterion': criterion,
              'hardware_accuracy_status': 'not_established'}
    if a.shape != e.shape or not a.size:
        result.update(comparable=False, criterion_passed=False if criterion else None,
                      reason='shape mismatch or empty tensor')
        return result
    a, e = a.astype(np.float64), e.astype(np.float64)
    fa, fe = np.isfinite(a), np.isfinite(e)
    finite = bool(fa.all() and fe.all())
    result.update(comparable=True, elements=a.size, finite=finite,
                  actual_nonfinite=int((~fa).sum()), expected_nonfinite=int((~fe).sum()))
    def scale(x, valid):
        if not valid.all():return None
        return {'max_abs': float(np.abs(x).max()), 'rms': float(np.sqrt(np.mean(x*x))),
                'l2': float(np.linalg.norm(x.ravel())), 'mean': float(x.mean())}
    result.update(actual_scale=scale(a,fa), expected_scale=scale(e,fe))
    if not finite:
        finite_pairs = fa & fe
        violations = ~finite_pairs
        if criterion is not None:
            violations[finite_pairs] = np.abs(a[finite_pairs]-e[finite_pairs]) > atol+rtol*np.abs(e[finite_pairs])
        result.update(max_abs=None, mean_abs=None, percentiles_abs=None, relative_l2=None,
                      cosine_similarity=None, criterion_passed=False if criterion else None,
                      violation_count=int(violations.sum()) if criterion else None,
                      violation_fraction=float(violations.mean()) if criterion else None,
                      violation_count_scope='nonfinite pairs plus finite elementwise threshold violations')
        return result
    delta=a-e; absolute=np.abs(delta); an=np.linalg.norm(a.ravel()); en=np.linalg.norm(e.ravel()); dn=np.linalg.norm(delta.ravel())
    result.update(max_abs=float(absolute.max()),mean_abs=float(absolute.mean()),
                  percentiles_abs={str(p):float(np.percentile(absolute,p)) for p in (50,90,95,99,99.9)},
                  relative_l2=float(dn/en) if en else (0.0 if not an else None),
                  relative_l2_note='undefined for nonzero actual against zero reference' if not en and an else None,
                  cosine_similarity=float(np.clip(np.dot(a.ravel(),e.ravel())/(an*en),-1,1)) if an and en else None,
                  cosine_note='undefined if either vector is zero' if not an or not en else None,
                  value_exact=bool(np.array_equal(a,e)))
    if criterion is None:
        result.update(criterion_passed=None,violation_count=None,violation_fraction=None)
    else:
        atol,rtol=criterion['atol'],criterion['rtol']
        if not np.isfinite(atol) or not np.isfinite(rtol) or min(atol,rtol)<0:raise ValueError('invalid criterion')
        violations=absolute>atol+rtol*np.abs(e)
        result.update(criterion_passed=not bool(violations.any()),violation_count=int(violations.sum()),
                      violation_fraction=float(violations.mean()))
    return result


def selection_properties(selected, position, context, valid=None, block_size=128, count=16):
    ids=np.asarray(selected,dtype=np.int64).reshape(-1)
    if valid is not None:ids=ids[np.asarray(valid,dtype=bool).reshape(-1)]
    current=position//block_size
    errors=[]
    if (ids<0).any() or (ids>=context//block_size).any():errors.append('out-of-context block')
    if (ids>current).any():errors.append('future block')
    if np.sum(ids==current)!=1:errors.append('current block must occur exactly once')
    if len(set(ids.tolist()))!=len(ids):errors.append('duplicate valid block')
    if len(ids)!=min(count,current+1):errors.append('wrong number of valid blocks')
    return {'passed':not errors,'errors':errors,'valid_block_ids':ids.tolist()}


def topk_boundary(actual_scores, reference_scores, actual_ids, reference_ids, eligible, k=15):
    """Report observed ambiguity, not an acceptance rule for a quantized model."""
    a,e=numpy_value(actual_scores).reshape(-1),numpy_value(reference_scores).reshape(-1)
    eligible=np.asarray(eligible,dtype=bool).reshape(-1)
    if a.shape!=e.shape or eligible.shape!=e.shape:raise ValueError('score shape mismatch')
    if not (np.isfinite(a[eligible]).all() and np.isfinite(e[eligible]).all()):raise ValueError('nonfinite eligible scores')
    if k <= 0:raise ValueError('past k must be positive')
    ids=np.flatnonzero(eligible);order=ids[np.argsort(-e[ids],kind='stable')]
    aa=set(np.asarray(actual_ids,dtype=int).reshape(-1).tolist());rr=set(np.asarray(reference_ids,dtype=int).reshape(-1).tolist())
    aa={i for i in aa if 0<=i<len(e) and eligible[i]};rr={i for i in rr if 0<=i<len(e) and eligible[i]}
    perturb=float(np.max(np.abs(a[eligible].astype(float)-e[eligible].astype(float)),initial=0))
    margin=float(e[order[k-1]]-e[order[k]]) if len(order)>k else None
    kth=float(e[order[k-1]]) if len(order)>=k else None
    changed=sorted(aa^rr)
    def admissible(scores, selected):
        if len(selected) != min(k, len(ids)):return False
        omitted = set(ids.tolist()) - selected
        return not selected or not omitted or min(float(scores[i]) for i in selected) >= max(float(scores[i]) for i in omitted)
    return {'past_k':k,'boundary_margin':margin,'kth_score':kth,
            'exact_kth_tie_count':int(np.sum(e[eligible]==kth)) if kth is not None else None,
            'observed_max_score_difference':perturb,
            'strict_gap_exceeds_twice_observed_difference':margin is not None and margin>2*perturb,
            'actual_set_admissible_for_own_scores':admissible(a, aa),
            'reference_set_admissible_for_own_scores':admissible(e, rr),
            'same_past_set':aa==rr,'only_actual':sorted(aa-rr),'only_reference':sorted(rr-aa),
            'exchanged_scores':{str(i):{'actual':float(a[i]),'reference':float(e[i])} for i in changed},
            'interpretation':'gap/perturbation localizes sensitivity; neither ties nor a small gap automatically justify changed selections'}


def assess_hardware_accuracy(capture):
    """Fail closed: no reviewed full-MSA FP16/MXFP6 policy is implemented.

    Historical `status=passed` refers to a CPU-derived provisional threshold.
    Neither that status nor a caller-supplied policy name is hardware acceptance.
    """
    return {'status': 'unverified', 'accepted': False,
            'reason': 'No reviewed full-MSA FP16/MXFP6 accuracy policy or validated precision-aware reference is implemented',
            'provisional_status': capture.get('provisional_status', capture.get('status', 'unknown'))}


def export_weight_identity(model, onnx_path):
    import onnx
    from onnx import numpy_helper
    graph=onnx.load(onnx_path).graph
    inits={i.name:numpy_helper.to_array(i) for i in graph.initializer}
    hashes={n:fingerprint(v) for n,v in inits.items() if np.issubdtype(v.dtype,np.floating)}
    result={}
    for name,value in model.state_dict().items():
        if not name.endswith('weight'):continue
        candidates=[('identity',value)]
        if value.ndim==2:candidates.append(('transpose',value.T))
        matches=[]
        for transform,tensor in candidates:
            h=fingerprint(tensor)
            matches.extend({'initializer':n,'transform':transform,'fingerprint':h} for n,v in hashes.items() if h==v)
        result[name]={'source':fingerprint(value),'matches':matches}
    return {'all_weight_tensors_matched':all(v['matches'] for v in result.values()),'weights':result,
            'floating_initializers':hashes,'custom_functions':[{'domain':f.domain,'name':f.name} for f in onnx.load(onnx_path,load_external_data=False).functions],
            'scope':'exact ONNX initializer values, accounting for transposes/dedup; not compiled MXFP6 weight identity'}
