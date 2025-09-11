import os

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def df_cleaner(df: pd.DataFrame, columns) -> pd.DataFrame:
    df = df[columns]
    missing_counts = df.isna().sum()
    print("Missing values in each column:")
    print(missing_counts)

    rows_before = df[columns].shape[0]
    print(f"Number of rows before dropping NaN: {rows_before}")

    cleaned_df = df[columns].dropna()

    rows_after = cleaned_df.shape[0]
    print(f"Number of rows after dropping NaN: {rows_after}")
    return cleaned_df.reset_index(drop=True)


def check_unique_per_subject(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Check if the specified column has a unique value for each PTID across all VISCODE2.
    Returns a DataFrame with PTID and the count of unique values in the specified column.
    """
    unique_check = df.groupby("PTID")[column].nunique().reset_index(name=f"{column}_unique_count")
    # print(f"\nChecking if {column} is unique for each PTID across visits:")
    # print(unique_check)

    # Identify PTIDs with non-unique values
    non_unique = unique_check[unique_check[f"{column}_unique_count"] > 1]
    if non_unique.empty:
        print(f"All PTIDs have a unique {column} across all visits.")
    else:
        print(f"PTIDs with non-unique {column} values:")
        print(non_unique)


def run():
    demographic = pd.read_csv("dataset/csv/PTDEMOG.csv")
    columns = ["PTID", "VISCODE2", "PTGENDER", "PTDOB", "PTHAND", "PTMARRY", "PTEDUCAT"]
    cleaned_demographic = df_cleaner(demographic, columns)
    print("Number of subjects with demographic: ", len(cleaned_demographic['PTID'].unique()))

    check_unique_per_subject(cleaned_demographic, "PTGENDER")
    # check_unique_per_subject(cleaned_demographic, "PTDOB")
    # check_unique_per_subject(cleaned_demographic, "PTHAND")
    # check_unique_per_subject(cleaned_demographic, "PTMARRY")
    # check_unique_per_subject(cleaned_demographic, "PTEDUCAT")

    demographic_columns = ["PTGENDER", "PTDOB", "PTHAND", "PTMARRY", "PTEDUCAT"]
    demographic_modes = cleaned_demographic.groupby('PTID')[demographic_columns].agg(
        lambda x: x.value_counts().idxmax()
    ).reset_index()

    MMSE = pd.read_csv("dataset/csv/MMSE.csv")
    columns = ["PTID", "VISCODE2", "MMSCORE"]
    cleaned_MMSE = df_cleaner(MMSE, columns)
    print("Number of subjects with MMSCORE: ", len(cleaned_MMSE['PTID'].unique()))

    ADAS = pd.read_csv("dataset/csv/ADAS.csv")
    columns = ["PTID", "VISCODE2", "TOTSCORE", "TOTAL13"]
    cleaned_ADAS = df_cleaner(ADAS, columns)
    print("Number of subjects with ADAS: ", len(cleaned_ADAS['PTID'].unique()))

    FAQ = pd.read_csv("dataset/csv/FAQ.csv")
    columns = ["PTID", "VISCODE2", "FAQTOTAL"]
    cleaned_FAQ = df_cleaner(FAQ, columns)
    print("Number of subjects with FAQ: ", len(cleaned_FAQ['PTID'].unique()))

    DX = pd.read_csv("dataset/csv/DXSUM.csv")
    print("Number of subjects with diagnosis: ", len(DX['PTID'].unique()))
    columns = ["PTID", "VISCODE", "VISCODE2", "EXAMDATE", "DIAGNOSIS"]
    cleaned_DX = df_cleaner(DX, columns)

    merged_df = cleaned_DX.merge(
        cleaned_MMSE, on=["PTID", "VISCODE2"], how="left"
    ).merge(
        cleaned_ADAS, on=["PTID", "VISCODE2"], how="left"
    ).merge(
        cleaned_FAQ, on=["PTID", "VISCODE2"], how="left"
    )

    merged_df = merged_df.merge(demographic_modes, on="PTID", how="left")

    missing_counts = merged_df.isna().sum()
    print("Missing values in each column:")
    print(missing_counts)

    final_columns = ["PTID", "VISCODE", "VISCODE2", "EXAMDATE", "DIAGNOSIS", "MMSCORE", "TOTSCORE", "TOTAL13",
                     "FAQTOTAL", "PTGENDER", "PTDOB", "PTHAND", "PTMARRY", "PTEDUCAT"]
    cleaned_merged = df_cleaner(merged_df, final_columns)
    print("Number of finall subjects: ", len(cleaned_merged['PTID'].unique()))

    # Convert dates to datetime objects
    cleaned_merged['PTDOB'] = pd.to_datetime(cleaned_merged['PTDOB'] + '/15', format='%m/%Y/%d')
    cleaned_merged['EXAMDATE'] = pd.to_datetime(cleaned_merged['EXAMDATE'], format='%Y-%m-%d')

    # Calculate age in years using relativedelta
    cleaned_merged['AGE'] = cleaned_merged.apply(
        lambda row: (row['EXAMDATE'] - row['PTDOB']).days / 365.25, axis=1
    )

    # Display a sample to verify
    print("\nSample of the final DataFrame with AGE:")
    print(cleaned_merged[['PTID', 'VISCODE2', 'EXAMDATE', 'PTDOB', 'AGE']].head())

    # Print uniques for verification
    print("\nUnique values in key columns:")
    for col in ['DIAGNOSIS', 'PTGENDER', 'PTHAND', 'PTMARRY']:
        print(f"{col}: {cleaned_merged[col].unique()}")

    # Encode categoricals
    # Binary shift (assuming PTGENDER/PTHAND start at 1)
    cleaned_merged['PTGENDER'] = cleaned_merged['PTGENDER'] - 1
    cleaned_merged['PTHAND'] = cleaned_merged['PTHAND'] - 1

    # One-hot for nominal: PTMARRY
    ohe_cols = ['PTMARRY']

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe_encoded = ohe.fit_transform(cleaned_merged[ohe_cols])
    ohe_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(ohe_cols))
    cleaned_merged = pd.concat([cleaned_merged.drop(ohe_cols, axis=1), ohe_df], axis=1)

    os.makedirs('dataset/tabular/', exist_ok=True)
    cleaned_merged.to_csv(f'dataset/tabular/all.csv', index=False)

    # Define numerical cols for scaling (AGE now included)
    numerical_cols = ['MMSCORE', 'TOTSCORE', 'TOTAL13', 'FAQTOTAL', 'PTEDUCAT', 'AGE']

    # Features list (dynamic)
    features = numerical_cols + ['PTGENDER'] + ['PTHAND'] + list(ohe.get_feature_names_out(ohe_cols))

    # First split: 80% train, 20% temp (val + test)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, temp_idx = next(gss1.split(cleaned_merged, groups=cleaned_merged['PTID']))

    train = cleaned_merged.iloc[train_idx]

    # Second split: 50% of temp for val (10% overall), 50% for test (10% overall)
    temp_df = cleaned_merged.iloc[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx_rel, test_idx_rel = next(gss2.split(temp_df, groups=temp_df['PTID']))

    # Convert relative indices to absolute
    val_idx = temp_df.iloc[val_idx_rel].index
    test_idx = temp_df.iloc[test_idx_rel].index

    val = cleaned_merged.loc[val_idx]
    test = cleaned_merged.loc[test_idx]

    train = train.copy()
    val = val.copy()
    test = test.copy()

    # Verify shapes and unique PTIDs
    print(f"Train shape: {train.shape}, Unique PTIDs: {train['PTID'].nunique()}")
    print(f"Val shape: {val.shape}, Unique PTIDs: {val['PTID'].nunique()}")
    print(f"Test shape: {test.shape}, Unique PTIDs: {test['PTID'].nunique()}")

    # Check for overlap in PTIDs
    train_ptids = set(train['PTID'])
    val_ptids = set(val['PTID'])
    test_ptids = set(test['PTID'])
    print("Overlap train-val:", train_ptids.intersection(val_ptids))
    print("Overlap train-test:", train_ptids.intersection(test_ptids))
    print("Overlap val-test:", val_ptids.intersection(test_ptids))

    scaler = MinMaxScaler()
    scaler.fit(train[numerical_cols])
    train[numerical_cols] = scaler.transform(train[numerical_cols])
    val[numerical_cols] = scaler.transform(val[numerical_cols])
    test[numerical_cols] = scaler.transform(test[numerical_cols])

    # Save X and y for each split
    for split_name, df in [('train', train), ('val', val), ('test', test)]:
        x = df[features]
        y = df['DIAGNOSIS'] - 1

        # Save to CSV
        x.to_csv(f'dataset/tabular/{split_name}_x.csv', index=False)
        y.to_csv(f'dataset/tabular/{split_name}_y.csv', index=False, header=['DIAGNOSIS'])  # Include header for clarity

        # Print class distribution
        print(f"\n{split_name.capitalize()} class distribution:")
        print(y.value_counts())

    # print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
    # print("Class distribution in train:", np.bincount(y_train))
    # print("Class distribution in test:", np.bincount(y_test))
    # y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
