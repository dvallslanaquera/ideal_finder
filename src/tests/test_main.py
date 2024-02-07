import unittest
import pandas as pd
import numpy as np
from main import CSVDatabaseOperations

class TestCSVDatabaseOperations(unittest.TestCase):
    def setUp(self):
        self.db_ops = CSVDatabaseOperations()
        self.db_ops.load_csv_files_to_db('datasets/')
        self.train_df = self.db_ops.read_sql("train")
        self.test_df = self.db_ops.read_sql("test")
        self.ideal_df = self.db_ops.read_sql("ideal")

    def test_load_csv_files_to_db(self):
        # Check if CSV files are correctly loaded
        result = self.db_ops.load_csv_files_to_db("datasets/")
        self.assertTrue(result) 

    def test_dataframes_length(self):
        # Ensure that train_df, test_df, and ideal_df have the same length
        train_df = self.db_ops.read_sql("train")
        test_df = self.db_ops.read_sql("test")
        ideal_df = self.db_ops.read_sql("ideal")

        # Assert that the lengths are equal
        self.assertEqual(len(train_df), len(test_df))
        self.assertEqual(len(train_df), len(ideal_df))
        self.assertEqual(len(test_df), len(ideal_df))
        
    def test_rows_order(self):
        # Check if the order of the rows is the same in all files
        pd.testing.assert_index_equal(self.train_df.index, self.test_df.index)
        pd.testing.assert_index_equal(self.train_df.index, self.ideal_df.index)

    def test_missing_values(self):
        # Check for missing values
        self.assertFalse(self.train_df.isnull().values.any())
        self.assertFalse(self.test_df.isnull().values.any())
        self.assertFalse(self.ideal_df.isnull().values.any())

    def test_numerical_values(self):\
        # Check whether all values are numerical
        self.assertTrue(self.train_df.select_dtypes(include=[np.number]).shape[1] == self.train_df.shape[1])
        self.assertTrue(self.test_df.select_dtypes(include=[np.number]).shape[1] == self.test_df.shape[1])
        self.assertTrue(self.ideal_df.select_dtypes(include=[np.number]).shape[1] == self.ideal_df.shape[1])

    def test_values_sorted_correctly(self):
        # Check if the order of all values is the same
        for df in [self.train_df, self.test_df, self.ideal_df]:
            sorted_df = df.sort_values(by=df.columns[0])
            self.assertTrue(sorted_df.equals(df))
            
    def test_column_x_consistency(self):
        train_x = self.train_df["x"].reset_index(drop=True)
        test_x = self.test_df["x"].reset_index(drop=True)
        ideal_x = self.ideal_df["x"].reset_index(drop=True)

        self.assertTrue((train_x == test_x).all())
        self.assertTrue((train_x == ideal_x).all())

if __name__ == "__main__":
    unittest.main()