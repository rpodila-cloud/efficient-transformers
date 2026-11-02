import tempfile
import unittest
from dataclasses import replace
from ..model import Workload,Candidate
from ..validation import tiny_check,pack_history,unpack_history

class NumericalTests(unittest.TestCase):
    @staticmethod
    def workload(batch=1):
        return Workload(batch_size=batch,ctx_len=8192,start_pos_id=4095,hidden_size=32,num_heads=4,num_kv_heads=2,
                        head_dim=8,rotary_dim=8,index_n_heads=2,indexer_n_head=1,index_head_dim=8,dtype='float32')
    def test_repacking(self):
        import torch
        logical=torch.arange(4*2*1024*8).reshape(4,2,1024,8)
        for dp,cp,unit in ((2,4,128),(4,8,1),(1,2,128)):
            self.assertTrue(torch.equal(logical,unpack_history(pack_history(logical,dp,cp,unit),4,2,dp,cp,unit)))
    def test_cp_chunks_and_boundaries(self):
        for idx,att,chunks in ((2,2,1),(4,2,2),(2,4,4)):
            with self.subTest(index_cp=idx,main_cp=att,chunks=chunks):
                c=Candidate(msa_indexer_cp=idx,msa_attn_cp=att,num_kv_blocks=chunks,num_cores=2)
                self.assertEqual(tiny_check(self.workload(),c,positions=(127,128,129,4095,4096))['status'],'passed')
    def test_batch_dp(self):
        for batch,idp,adp in ((2,1,2),(2,2,1),(4,2,2),(4,4,4)):
            with self.subTest(batch=batch,idp=idp,adp=adp):
                c=Candidate(msa_indexer_dp=idp,msa_attn_dp=adp,msa_indexer_cp=2,msa_attn_cp=2,num_kv_blocks=2,num_cores=2)
                self.assertEqual(tiny_check(self.workload(batch),c)['status'],'passed')
    def test_decomposed_export(self):
        with tempfile.TemporaryDirectory() as d:
            c=Candidate(msa_indexer_cp=2,msa_attn_cp=2,num_kv_blocks=2,num_cores=2)
            self.assertTrue(tiny_check(self.workload(),c,positions=(4095,4096),export_dir=d)['onnx_parity'])
