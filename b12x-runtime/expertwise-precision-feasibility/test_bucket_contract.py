import unittest
from bucket_contract import banks, expanded_routes


class BucketTests(unittest.TestCase):
    def test_arbitrary_bucket_counts_partition_all_native_experts(self):
        b=banks([287])
        self.assertEqual(len(b[2]['original_ids']),287)
        self.assertEqual(len(b[3]['original_ids']),1)
        self.assertEqual(sorted(b[2]['original_ids']+b[3]['original_ids']),list(range(288)))

    def test_rankstack_contributions_preserve_original_router_weights(self):
        b=banks([1,5,287]); ids=[0,1,5,100,287]; weights=[.1,.2,.3,.4,.5]
        # Distinct expert/rank values catch permutation, duplication and /4 errors.
        expected=sum(w*sum(e*10+r for r in range(4)) for e,w in zip(ids,weights))
        actual=0
        for bank in b.values():
            routed=expanded_routes(bank,ids,weights)
            self.assertEqual(len(routed),len(ids)*4)
            sentinel=len(bank['original_ids'])*4
            for virtual,w in routed:
                if virtual==sentinel:
                    self.assertEqual(w,0)
                else:
                    original=bank['original_ids'][virtual//4]
                    actual+=w*(original*10+virtual%4)
        self.assertAlmostEqual(actual,expected)

    def test_empty_bucket_retains_single_launch_fastpath(self):
        self.assertEqual(set(banks([])),{2})
        self.assertEqual(set(banks(list(range(288)))),{3})

    def test_invalid_or_duplicate_original_ids_rejected(self):
        for ids in ([1,1],[2,1],[-1],[288],[True],[2.0]):
            with self.assertRaises(ValueError): banks(ids)
        with self.assertRaises(ValueError): banks([],287)

    def test_routes_reject_unknown_global_ids(self):
        b=banks([2])[2]
        for ids in ([-1],[288],[True]):
            with self.assertRaises(ValueError): expanded_routes(b,ids,[1])
        with self.assertRaises(ValueError): expanded_routes(b,[2],[1],2)


if __name__=='__main__': unittest.main()
