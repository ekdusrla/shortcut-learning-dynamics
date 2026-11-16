import unittest
import pandas as pd
from scripts.waterbirds_vgg50.analyze_history import derive,intervals


class Tests(unittest.TestCase):
    def test_constant_majority_classifier_is_not_core_learning(self):
        row=dict(val_g00_n=3,val_g01_n=3,val_g10_n=1,val_g11_n=1,
            val_g00_acc=1.,val_g01_acc=1.,val_g10_acc=0.,val_g11_acc=0.,
            val_acc=.75,val_wga=0.,val_aligned_acc=.75,val_conflicting_acc=.75,
            val_cf_acc=.75,val_cf_flip_rate=0.,val_cf_original_correct_n=6,
            val_cf_correct_to_incorrect_rate=0.)
        result=derive(pd.DataFrame([row])).iloc[0]
        self.assertEqual(result.val_balanced_acc,.5)
        self.assertEqual(result.val_balanced_conflicting_acc,.5)
        self.assertEqual(result.val_class1_recall,0.)
        self.assertEqual(result.val_c2i_unconditional,0.)
        for wrong in ({'val_acc':.8},{'val_wga':.1},{'val_cf_flip_rate':.2}):
            with self.assertRaises(ValueError):derive(pd.DataFrame([{**row,**wrong}]))

    def test_sustained_runs_do_not_join_spikes_or_missing_epochs(self):
        self.assertEqual(intervals([1,2,3,4,5,6,7],[False,True,True,False,True,True,True]),[(2,3),(5,7)])
        self.assertEqual(intervals([1,2,5,6],[True,True,True,True]),[(1,2),(5,6)])
        self.assertEqual(intervals([1,2,3],[False,False,False]),[])

if __name__=='__main__':unittest.main()
