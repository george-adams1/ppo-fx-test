import pandas as pd


def add_missing_date_symbol_pairs(df, date_column_name, symbol_column_name):
    """
    Add missing date and symbol pairs to the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with date and symbol columns. The date column should be in
        a format convertible to datetime.
    date_column_name : str
        Name of the column in df that contains the date information.
    symbol_column_name : str
        Name of the column in df that contains the symbol information.

    Returns
    -------
    df : pandas.DataFrame
        Updated DataFrame with missing (date, symbol) pairs added. Each missing pair
        is filled with zeros for all other columns.
    """

    # Convert date_column to datetime
    df[date_column_name] = pd.to_datetime(df[date_column_name])

    # Set date_column and symbol_column as the DataFrame index
    df.set_index([date_column_name, symbol_column_name], inplace=True)

    # Get all unique dates
    unique_dates = df.index.get_level_values(date_column_name).unique()

    # Get all unique symbols
    unique_symbols = df.index.get_level_values(symbol_column_name).unique()

    # Create a set of all possible (date, symbol) pairs
    all_pairs = {(date, symbol) for date in unique_dates for symbol in unique_symbols}

    # Create a set of all (date, symbol) pairs that are in the DataFrame's index
    existing_pairs = set(df.index)

    # Find the missing pairs
    missing_pairs = all_pairs - existing_pairs

    # Create a DataFrame with the missing rows (filled with 0)
    missing_rows = pd.DataFrame(0, index=pd.MultiIndex.from_tuples(missing_pairs, names=[date_column_name,
                                                                                         symbol_column_name]),
                                columns=df.columns)

    # Concatenate the original DataFrame with the missing rows
    df = pd.concat([df, missing_rows])

    df.reset_index(inplace=True)

    return df


def create_unix_date_nums(df):
    """
    Converts a DataFrame column 'StartTime' from string format to unix timestamp and stores it in a new column 'date_nums'.

    Parameters:
    df (pandas.DataFrame): The input DataFrame that contains a 'StartTime' column in string format.

    Returns:
    df (pandas.DataFrame): The same DataFrame that was passed in, but with an additional 'date_nums' column which contains the unix timestamp representation of 'StartTime'.
    """
    # Convert StartTime to datetime if it's not
    df['StartTime'] = pd.to_datetime(df['StartTime'])

    # Create a numeric representation of StartTime (in unix time)
    df['date_nums'] = (df['StartTime'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

    return df
