import copy
import unittest
from projection_contract import packed_layout, resolve_layer_projection_bits


class ProjectionTests(unittest.TestCase):
    def test_selected_down_only_no_config_mutation(self):
        plan={'5':{'down_proj':3},'32':{'down_proj':3}}
        before=copy.deepcopy(plan)
        for layer in range(3,45):
            expected=(2,2,3) if layer in (5,32) else (2,2,2)
            self.assertEqual(resolve_layer_projection_bits(2,plan,layer),expected)
        self.assertEqual(plan,before)

    def test_shape_delta_exactly_one_mib_per_expert(self):
        a,b=packed_layout((2,2,2)),packed_layout((2,2,3))
        self.assertEqual(a['shapes_per_rank']['w13_trellis'],b['shapes_per_rank']['w13_trellis'])
        self.assertEqual(b['shapes_per_rank']['w2_trellis'],(288,32,256,48))
        self.assertEqual(b['total_trellis_bytes']-a['total_trellis_bytes'],288*1024**2)
        self.assertEqual(6*(b['total_trellis_bytes']-a['total_trellis_bytes']),1811939328)

    def test_dynamic_kernel_dispatch_for_projection_tuple(self):
        self.assertEqual(packed_layout((2,2,3))['native_dispatch_k'],0)
        self.assertEqual(packed_layout((2,2,2))['native_dispatch_k'],2)
        self.assertEqual(packed_layout((3,3,3))['native_dispatch_k'],3)

    def test_reject_unsupported_gate_up_split_and_native_changes(self):
        for p in ({'5':{'gate_proj':3}}, {'45':{'down_proj':3}},
                  {'2':{'down_proj':3}}, {'5':{'lm_head':3}},
                  {'5':{'down_proj':True}}, {'05':{'down_proj':3}}):
            with self.assertRaises(ValueError):resolve_layer_projection_bits(2,p,5)
        with self.assertRaises(ValueError):packed_layout((2,3,3))
        with self.assertRaises(ValueError):packed_layout((2,2,3),experts=287)
        with self.assertRaises(ValueError):resolve_layer_projection_bits(2,{},45)


if __name__=='__main__':unittest.main()
