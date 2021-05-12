"""Unittest stats.py"""
import unittest
import numpy as np
from craterpy import stats as cs


class Test_quickstats_helpers(unittest.TestCase):
    """Test quickstats.py helper functions"""

    def test_list_quickstats(self):
        """Test the _listStats function"""
        expected = np.array(["maximum", "mean", "median"])
        actual = cs._list_quickstats()[:3]
        np.testing.assert_array_equal(actual, expected)

    def test_get_quickstats_functions_single(self):
        """Test the _get_quickstats_functions function with one stat"""
        expected_name = "maximum"
        stat_func = cs._get_quickstats_functions("maximum")
        actual_name = stat_func[0][0]
        actual_func = stat_func[0][1]
        self.assertEqual(actual_name, expected_name)
        self.assertEqual(callable(actual_func), True)

    def test_get_quickstats_functions_multi(self):
        """Test the _get_quickstats_functions function with list of stats"""
        expected_names = ["maximum", "mean"]
        stats_funcs = cs._get_quickstats_functions(["maximum", "mean"])
        actual_names = [pair[0] for pair in stats_funcs]
        funcs_callable = [callable(pair[1]) for pair in stats_funcs]
        self.assertEqual(actual_names, expected_names)
        self.assertEqual(funcs_callable, [True, True])

    def test_get_quickstats_functions_undefined(self):
        """Test that undefined stat raises Exception"""
        self.assertRaises(ValueError, cs._get_quickstats_functions, "NotAStat")


# # Test Compute Stats
# class Test_compute_stats(unittest.TestCase):
#     """Test computeStats function"""

#     def setUp(self):
#         self.data_path = p.join(craterpy.__path__[0], "data")
#         self.crater_csv = p.join(self.data_path, "craters.csv")
#         self.moon_tif = p.join(self.data_path, "moon.tif")
#         self.df = pd.read_csv(self.crater_csv)
#         self.cds = CraterpyDataset(self.moon_tif, radius=1737)

# def test_one_crater_one_stat(self):
#     """Test mean on first crater in df"""
#     # af.compute_stats(self.df, self.cds, 'mean', self.cdf.index[0:5])
#     # TODO: implement
#     pass

# def test_one_crater_many_stats(self):
#     pass

# def test_many_craters_many_stats(self):
#     pass

# def test_crater_stats(self):
#     pass

# def test_ejecta_stats(self):
#     pass
