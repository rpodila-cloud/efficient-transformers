"""Pure candidate generation and logical estimates; never loads a compiler/device."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
import json
import math

POLICY_VTCM = 12 * 1024**2
PHYSICAL_VTCM = 16 * 1024**2
IMPL = "msa_indexer_attention_GP_unified"


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Workload:
    batch_size: int = 1
    ctx_len: int = 1048576
    start_pos_id: int = 1048571
    hidden_size: int = 6144
    num_heads: int = 64
    num_kv_heads: int = 4
    head_dim: int = 128
    rotary_dim: int = 64
    rope_theta: float = 5000000.0
    index_n_heads: int = 4
    indexer_n_head: int = 1
    index_head_dim: int = 128
    index_block_size: int = 128
    index_topk_blocks: int = 16
    dtype: str = "float16"
    kv_cache_dtype: str = "float16"
    mxfp6: bool = True
    onnx_optimize: bool = True
    seed: int = 123

    def validate(self):
        for k, v in asdict(self).items():
            if k not in {"dtype", "kv_cache_dtype", "mxfp6", "onnx_optimize", "rope_theta"}:
                integer(v, k, 0 if k in {"start_pos_id", "seed"} else 1)
        if type(self.rope_theta) not in (float, int) or not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")
        if type(self.mxfp6) is not bool or type(self.onnx_optimize) is not bool:
            raise ValueError("mxfp6 and onnx_optimize must be booleans")
        if self.dtype not in {"float16", "float32", "bfloat16"} or self.kv_cache_dtype not in {"float16", "bfloat16"}:
            raise ValueError("supported model types: float16/float32/bfloat16; cache: float16/bfloat16")
        if self.kv_cache_dtype == "bfloat16" and self.dtype != "bfloat16":
            raise ValueError("BF16 cache requires BF16 model (bench.py)")
        if self.index_block_size != 128 or self.index_topk_blocks != 16:
            raise ValueError("this configurator preserves block size 128 and selection of 16 blocks")
        if self.ctx_len < 2048 or self.ctx_len % 128:
            raise ValueError("context must contain at least 16 complete 128-token blocks")
        if self.seed != 123:
            raise ValueError("benchmark fixes model seed123 and input seed17; seed tuning is unsupported")
        if self.start_pos_id >= self.ctx_len:
            raise ValueError("position must be inside the requested context")
        if self.num_heads % self.num_kv_heads or self.index_n_heads % self.indexer_n_head:
            raise ValueError("query heads must divide into their KV head groups")
        if self.index_n_heads != self.num_kv_heads:
            raise ValueError("unified selector requires index Q heads == main KV heads (forwards.py:2587)")
        if self.rotary_dim % 2 or self.rotary_dim > min(self.head_dim, self.index_head_dim):
            raise ValueError("rotary_dim must be even and fit both head dimensions")


@dataclass(frozen=True)
class Resources:
    num_soc: int = 16
    cores_per_soc: int = 4
    ddr_bytes: int | dict[str, int] | None = None
    ddr_headroom_bytes: int = 0
    physical_vtcm_bytes: int = PHYSICAL_VTCM
    vtcm_budget_bytes: int = POLICY_VTCM

    def validate(self):
        for k in ("num_soc", "cores_per_soc", "physical_vtcm_bytes", "vtcm_budget_bytes"):
            integer(getattr(self, k), k)
        integer(self.ddr_headroom_bytes, "ddr_headroom_bytes", 0)
        if self.cores_per_soc > 4:
            raise ValueError("AI200 exposes at most four cores per SoC")
        if self.physical_vtcm_bytes > PHYSICAL_VTCM:
            raise ValueError("physical capacity exceeds inspected AI200 capacity")
        if self.vtcm_budget_bytes > min(POLICY_VTCM, self.physical_vtcm_bytes):
            raise ValueError("VTCM policy cannot exceed 12 MiB or physical capacity")
        if self.ddr_bytes is not None:
            values = self.ddr_bytes.values() if isinstance(self.ddr_bytes, dict) else [self.ddr_bytes]
            for v in values:
                integer(v, "ddr_bytes")
            if isinstance(self.ddr_bytes, dict) and not self.ddr_bytes:
                raise ValueError("empty per-device DDR budgets")

    def budget(self, qid):
        if isinstance(self.ddr_bytes, dict):
            return self.ddr_bytes.get(str(qid))
        return self.ddr_bytes


@dataclass(frozen=True)
class Candidate:
    msa_indexer_dp: int = 1
    msa_indexer_cp: int = 16
    msa_attn_dp: int = 1
    msa_attn_cp: int = 4
    num_kv_blocks: int = 8
    indexer_num_blocks: int | None = None
    msa_num_kv_blocks: int | None = None
    num_cores: int = 4

    @property
    def chunks(self):
        return self.num_kv_blocks if self.indexer_num_blocks is None else self.indexer_num_blocks

    def effective(self):
        return {k: v for k, v in asdict(self).items()
                if k not in {"num_kv_blocks", "indexer_num_blocks", "msa_num_kv_blocks"}} | {"indexer_chunks": self.chunks}

    def key(self, workload):
        return digest({"workload": asdict(workload), "effective": self.effective(), "impl": IMPL})[:20]

    def reasons(self, w, r):
        errors = []
        try:
            w.validate(); r.validate()
            for k, v in asdict(self).items():
                if v is not None or k not in {"indexer_num_blocks", "msa_num_kv_blocks"}:
                    integer(v, k)
        except ValueError as e:
            return [str(e)]
        for stage in ("indexer", "attn"):
            dp, cp = getattr(self, f"msa_{stage}_dp"), getattr(self, f"msa_{stage}_cp")
            if w.batch_size % dp:
                errors.append(f"batch_size must be divisible by msa_{stage}_dp (forwards.py:2305,2479)")
            if cp <= 1:
                errors.append(f"msa_{stage}_cp must exceed one (forwards.py:2283)")
        if self.num_cores > r.cores_per_soc:
            errors.append("requested cores exceed available cores per SoC")
        alignment = self.msa_indexer_cp * self.chunks * self.num_cores * w.index_block_size
        if w.ctx_len % alignment:
            errors.append("context must divide indexCP*chunks*cores*128 exactly (forwards.py:2306–2310,2423)")
        if w.ctx_len % self.msa_attn_cp or w.index_block_size % self.msa_attn_cp:
            errors.append("main CP must divide context and 128-token blocks (forwards.py:2573,2608)")
        selected = w.index_topk_blocks * w.index_block_size
        if selected % self.num_cores:
            errors.append("selected tokens must divide across cores (forwards.py:2583)")
        local_blocks = w.ctx_len // (self.msa_indexer_cp * self.num_cores * w.index_block_size)
        if self.msa_indexer_cp * self.num_cores * min(w.index_topk_blocks, local_blocks) < w.index_topk_blocks:
            errors.append("insufficient candidates for global TopK (forwards.py:2470,2531)")
        return errors


def generate(w, r, ranges=None, maximum=64):
    """Returns rejected requests too. Limit enumeration before constructing a product."""
    integer(maximum, "maximum candidates")
    ranges = ranges or {}
    if set(ranges) - set(Candidate.__dataclass_fields__):
        raise ValueError(f"unknown candidate axes: {set(ranges) - set(Candidate.__dataclass_fields__)}")
    if any(not isinstance(v, list) or not v for v in ranges.values()):
        raise ValueError("candidate ranges must be nonempty lists")
    count = math.prod(len(v) for v in ranges.values())
    if count > maximum:
        raise ValueError(f"{count} requested combinations exceed maximum {maximum}")
    seen, rows = {}, []
    for vals in product(*ranges.values()):
        c = Candidate(**dict(zip(ranges, vals)))
        errors = c.reasons(w, r)
        key = c.key(w)
        row = {"id": key, "requested": asdict(c), "effective": c.effective(), "reasons": errors,
               "semantic_status": "rejected" if errors else "source_valid",
               "placement_status": "unverified", "ddr_status": "unverified", "vtcm_status": "unverified",
               "correctness_status": "unverified", "latency_status": "unmeasured",
               "inactive": {"msa_num_kv_blocks": "no consumer in unified decode"}}
        if not errors:
            if key in seen:
                row["duplicate_of"] = key
            else:
                row["estimates"] = estimates(w, c)
                seen[key] = True
        rows.append(row)
    return rows


def estimates(w, c):
    """Logical storage inventory, not a compiled simultaneous working-set prediction."""
    b, t, d, e, h, v, j = w.batch_size, w.ctx_len, w.head_dim, w.index_head_dim, w.hidden_size, w.num_kv_heads, w.index_n_heads
    ir = c.msa_indexer_dp * c.msa_indexer_cp * w.indexer_n_head
    ar = c.msa_attn_dp * c.msa_attn_cp * v
    tok = t // c.msa_indexer_cp // c.chunks // c.num_cores
    z = w.index_topk_blocks * w.index_block_size
    cache = {"index_k": b*t*w.indexer_n_head*e*2, "main_k": b*t*v*d*2, "main_v": b*t*v*d*2}
    weight_elements = h*(2*w.num_heads*d + 2*v*d + c.msa_indexer_dp*(j+w.indexer_n_head)*e) + 2*d + 2*e
    stages = []
    def add(stage, name, shape, itemsize, role, producer, consumers):
        stages.append(dict(stage=stage, tensor=name, logical_shape=shape, logical_bytes=math.prod(shape)*itemsize,
                           logical_itemsize=itemsize, role=role, producer=producer, consumers=consumers,
                           residency="unknown until compile", evidence="source estimate"))
    add("index_projection", "index_q", [b,j,e], 2 if w.dtype != "float32" else 4, "projection", "index_q_proj/norm/RoPE", ["index_score"])
    add("index_cache_score", "index_cache", [b//c.msa_indexer_dp,ir,t//c.msa_indexer_cp,e], 2, "retained cache", "owner-only scatter", ["chunk gather"])
    add("index_cache_score", "index_gather_core", [1,ir,c.num_cores,tok,e], 2, "gather", "cache", ["index_matmul"])
    add("index_cache_score", "index_scores", [1,ir,c.num_cores,j//w.indexer_n_head,tok], 4, "eager FP32 math", "index_matmul", ["max128"])
    k = min(w.index_topk_blocks,t//c.msa_indexer_cp//c.num_cores//128)
    add("local_selection", "core_candidates_scores_and_ids", [1,ir,c.num_cores,j//w.indexer_n_head,k], 12, "float32+int64 logical payload", "local_topk", ["global_topk"])
    add("main_projection", "qkv", [b,w.num_heads+2*v,d], 2 if w.dtype != "float32" else 4, "projection", "q/k/v_proj/norm/RoPE", ["main_cache", "attention"])
    add("global_selection", "global_topk_input", [b,max(ir,j),c.msa_indexer_cp*c.num_cores*k], 4, "logical candidate score replication", "core_candidates", ["global_topk/current_block"])
    for name in ("main_k", "main_v"):
        add("main_cache_gather", name, [b//c.msa_attn_dp,ar,t//c.msa_attn_cp,d], 2, "retained cache", "owner-only scatter", ["selected gather"])
        add("main_cache_gather", name+"_selected", [1,ar,c.num_cores,z//c.num_cores,d], 2, "masked gathered slots", name, ["attention"])
    add("attention", "scores_or_exp", [1,ar,c.num_cores,w.num_heads//v,z//c.num_cores], 4, "eager math; overlapping lifetimes unknown", "QK", ["exp/sum/weightedV"])
    add("merges", "partial_outputs", [1,ar,c.num_cores,w.num_heads//v,d], 4, "core then CP reduction", "weightedV", ["output_projection"])
    add("output_projection", "output", [b,1,h], 2 if w.dtype != "float32" else 4, "output", "o_proj", ["host"])
    return {"evidence": "logical estimates; not feasibility", "cache_bytes": cache, "total_cache_bytes": sum(cache.values()),
            "unpacked_weight_bytes": weight_elements*(4 if w.dtype == "float32" else 2),
            "mandatory_cache_write_payload_bytes": b*(2*v*d+w.indexer_n_head*e)*2,
            "index_scan_payload_bytes": cache["index_k"], "index_tokens_per_core_chunk": tok,
            "main_selected_slots_per_core": z//c.num_cores, "stages": stages,
            "ownership": {"index_k": "owner=(p//128)%indexCP; slot=(p//128//indexCP)*128+p%128", "main_kv": "owner=p%mainCP; slot=p//mainCP"},
            "unknown": ["compiled weight packing", "alignment", "peak overlapping live tensors", "kernel scratch", "communication staging and physical traffic", "actual device placement"],
            "main_blocking": "fixed selected_tokens/cores; msa_num_kv_blocks inactive"}
