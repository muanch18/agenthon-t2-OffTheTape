import unittest

from experiments.direct_candidate_comparison import summarize


class DirectComparisonTests(unittest.TestCase):
    def test_constant_paired_difference(self):
        rows = [dict(case_id=str(i), asof=str(i), seed=s,
                     pca_candidate=2., gaussian_candidate=1.)
                for i in range(6) for s in (17, 83, 2026)]
        result = summarize(rows, reps=100)
        self.assertEqual(result['delta_gaussian_minus_pca'], -1.)
        self.assertEqual(result['exploratory_95_interval'], [-1., -1.])
        self.assertEqual(result['cases'], 6)

    def test_seed_count_does_not_reweight_cases(self):
        rows = [dict(case_id='a', asof='1', seed=s,
                     pca_candidate=2., gaussian_candidate=1.) for s in (17, 83, 2026)]
        rows.append(dict(case_id='b', asof='2', seed=17,
                         pca_candidate=2., gaussian_candidate=3.))
        result = summarize(rows, reps=100)
        self.assertEqual(result['delta_gaussian_minus_pca'], 0.)
        self.assertEqual(result['gaussian_case_win_fraction'], .5)


if __name__ == '__main__':
    unittest.main()
