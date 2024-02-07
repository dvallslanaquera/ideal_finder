import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


# Custom exceptions definitions
class DatabaseConnectionError(Exception):
    """Raised when there's an issue connecting to the database."""

    pass


class DataLoadingError(Exception):
    """Raised when there's an error loading data into the database."""

    pass


class DataProcessingError(Exception):
    """Raised during errors in data processing."""

    pass


# Base class for database operations
class DatabaseOperations:
    def __init__(self, db_name: str = "sqlite:///ideal.db"):
        self.engine = create_engine(db_name)

    def read_sql(self, table_name: str):
        try:
            return pd.read_sql(table_name, self.engine)
        except Exception as e:
            print(f"Error reading from database: {e}")
            return None


# Example of inheritance creating a child class derived from DatabaseOperations
class CSVDatabaseOperations(DatabaseOperations):
    """
    Reads CSV files into a database using SQLAlchemy. It works with SQLite.

    Args:
        dir_path (str): Path to CSV files.
        db_name (str): SQLAlchemy database connection string.

    Returns:
        bool: True if successful, False otherwise.
    """

    def __init__(self, db_name: str = "sqlite:///ideal.db"):
        super().__init__(db_name)

    def load_csv_files_to_db(self, dir_path: str) -> bool:
        try:
            expected_files = ["ideal.csv", "test.csv", "train.csv"]
            for filename in expected_files:
                file_path = os.path.join(dir_path, filename)
                if os.path.exists(file_path):
                    data = pd.read_csv(file_path)
                    table_name = os.path.splitext(filename)[0]
                    data.to_sql(
                        table_name, self.engine, if_exists="replace", index=False
                    )
                    print("Data loaded successfully")
                else:
                    raise FileNotFoundError(f"File not found: {file_path}")
            return True
        except FileNotFoundError as e:
            print(e)
            return False
        except Exception as e:
            print(f"An error occurred while loading data: {e}")
            raise DataLoadingError


def calculate_least_squares(
    train: pd.DataFrame(), ideal: pd.DataFrame()
) -> pd.DataFrame():
    """
    Returns a dataframe containing the columns in train with the lowest deviation.

    Args:
        train (DataFrame)
        ideal (DataFrame)

    Returns:
        bool: True if successful, False otherwise.
    """
    res_data = []
    for col_train in train.iloc[:, 1:].columns:
        train_data = train[col_train].values
        ideal_data = ideal.iloc[:, 1:].values

        # Calculate the least squares deviation for all pairs
        deviations = np.sum((ideal_data - train_data[:, np.newaxis]) ** 2, axis=0)

        # Find the index and value of the minimum deviation
        idx = np.argmin(deviations) + 1
        ls = deviations[idx - 1]  # Subtract 1 to get the correct index
        res_data.append([idx, ls])
    return pd.DataFrame(res_data, columns=["index", "ls"])


def plot_data(train, ideal, ideal_columns):
    """
    Creates one image with 4 subplots. Each subplot pairs one column from 'train' with
    one column from 'ideal' as specified by the ideal_columns list.

    :param train: DataFrame containing train data.
    :param ideal: DataFrame containing ideal data.
    :param ideal_columns: List of column names from 'ideal' to plot.
    """
    plt.figure(figsize=(20, 10))

    # Create 4 subplots (2 rows, 2 columns)
    for i, ideal_col in enumerate(ideal_columns, start=1):
        plt.subplot(2, 2, i)
        train_col = train.columns[i]  # Corresponding column in 'train'
        plt.plot(train.index, train[train_col], label=f"Train: {train_col}", marker="o")
        plt.plot(ideal.index, ideal[ideal_col], label=f"Ideal: {ideal_col}", marker="x")
        plt.xlabel(xlabel="X")
        plt.ylabel(ylabel="Y")
        plt.title(label=f"Train: {train_col} vs Ideal: {ideal_col}")
        plt.legend()

    plt.tight_layout()
    plt.show()


def merge_and_trim_dataframes(
    ideal: pd.DataFrame, test: pd.DataFrame, train: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    The order and number of rows in test.csv can be mismatched when comparing it
    to train.csv and ideal.csv. This function aims to match the dataframes so the
    data is ready to be used.

    Args:
        ideal (pd.DataFrame)
        test (pd.DataFrame)

    Returns:
        DataFrame: returns both dataframes properly trimmed.
    """
    # Sort indeces in test dataset
    test = test.sort_values(by="x", ascending=True).reset_index(drop=True)

    # Drop duplicates
    test_trim = test.drop_duplicates(subset="x")

    # Extract the values from the "x" column in the "ideal" DataFrame
    common_x_values = ideal[ideal["x"].isin(test_trim["x"])]["x"]

    # Filter both dataframes to these values
    test_trim = test_trim[test_trim["x"].isin(common_x_values)].reset_index(drop=True)
    ideal_filtered = ideal[ideal["x"].isin(common_x_values)].reset_index(drop=True)
    train_filtered = train[train["x"].isin(common_x_values)].reset_index(drop=True)

    return ideal_filtered, test_trim, train_filtered


def final_plot(
    test_trim: pd.DataFrame, ideal_filtered: pd.DataFrame, ideal_col_index: list
):
    """
    Plots the test data against the ideal data for specified columns.

    Args:
        test_trim (pd.DataFrame): DataFrame containing trimmed test data.
        ideal_filtered (pd.DataFrame): DataFrame containing filtered ideal data.
        ideal_col_index (list): List of column names from ideal data to plot.
    """
    # Extract the X and Y data for "test"
    x_test = test_trim["x"]
    y_test = test_trim["y"]

    # Extract the X and Y data for "ideal_trim" columns
    ideal_data = {col: ideal_filtered[col] for col in ideal_col_index}

    # Create a new figure for the plot
    plt.figure()

    # Plot Y data for test
    plt.plot(x_test, y_test, label="test")

    # Plot Y data for each ideal_trim
    for col, y_data in ideal_data.items():
        plt.plot(x_test, y_data, label=f"{col} (ideal_trim)")

    # Setting labels and title
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Y vs X")

    # Show the plot
    plt.show()


def find_ideal(train_df, test_df, ideal_df) -> None:
    """
    Process data and perform analysis, including checking deviations.

    Args:
        train_df (pd.DataFrame): DataFrame containing training data.
        test_df (pd.DataFrame): DataFrame containing test data.
        ideal_df (pd.DataFrame): DataFrame containing ideal data.
        ideal_col_index (List[str]): List of column names from ideal data for analysis.
    """
    # Calculate least squares to find ideal values
    res = calculate_least_squares(train_df, ideal_df)
    print("Least squares calculated!")
    print(res)

    # Select columns to be printed
    ideal_col_index = ["y" + str(int(idx)) for idx in res.iloc[:, 0]]
    plot_data(train=train_df, ideal=ideal_df, ideal_columns=ideal_col_index)

    # Match both test and ideal dataframes
    ideal_trim, test_trim, train_trim = merge_and_trim_dataframes(
        ideal=ideal_df, test=test_df, train=train_df
    )
    print("Dataframes merged and trimmed successfully")

    # Define factor to check the deviation
    factor = np.sqrt(2)
    idx = 1
    fitting_functions = []

    # Check deviations for each ideal function
    for col_idx in range(1, 4):
        colname = ideal_col_index[col_idx]

        # Calculate maximum deviation between ideal.csv and train.csv
        previous_dev = np.max((ideal_trim[colname] - train_trim.iloc[:, idx]) ** 2)

        # Calculate maximum deviation between test.csv and ideal.csv
        current_dev = np.max((ideal_trim[colname] - test_trim.iloc[:, 1]) ** 2)

        # Assign criterion to check deviations
        exceeds = current_dev > (previous_dev * factor)

        # Check and print results
        results_map = np.where(exceeds, "Exceeds criterion", "Meets criterion")
        print(f"Results for ideal function {ideal_col_index[idx]}: {results_map}")

        # Add to list if meets criterion
        if not exceeds:
            fitting_functions.append(ideal_col_index[idx])
        idx += 1

    # Print the functions that meet the criterion
    if fitting_functions:
        print("Functions that meet the criterion:", fitting_functions)
    else:
        print("No functions met the criterion")

    # Call the final plotting function
    final_plot(
        test_trim=test_trim,
        ideal_filtered=ideal_trim,
        ideal_col_index=ideal_col_index
    )


# Main script workflow
if __name__ == "__main__":
    try:
        curr_path = os.getcwd()
        db_operations = CSVDatabaseOperations("sqlite:///ideal.db")
        if not db_operations.load_csv_files_to_db(
            dir_path=os.path.join(curr_path, "./datasets")
        ):
            raise DataLoadingError("Error loading CSV files into the database")

        train_df = db_operations.read_sql("train")
        ideal_df = db_operations.read_sql("ideal")
        test_df = db_operations.read_sql("test")
        if train_df is None or ideal_df is None or test_df is None:
            raise DataProcessingError("Error reading data from database")

        find_ideal(train_df=train_df, test_df=test_df, ideal_df=ideal_df)

    except (DatabaseConnectionError, DataLoadingError, DataProcessingError) as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)


