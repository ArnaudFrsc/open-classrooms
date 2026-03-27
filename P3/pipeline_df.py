import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None

########################
######## UTILS #########
########################

def remove_outlier_rows(df, columns, factor=1.5):
    """
    Removes rows that are outliers in the specified numeric columns,
    but keeps rows where those columns have NaN values.

    params:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to check for outliers.
        factor (float): The IQR multiplier for defining outlier bounds (default is 1.5).

    returns:
        pd.DataFrame: A DataFrame with outliers removed, but NaNs preserved.
    """
    mask = pd.Series(True, index=df.index)

    for col in columns:
        if col not in df.columns:
            continue  # Skip missing columns safely

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        # Keep rows that are within bounds OR NaN
        mask &= df[col].between(lower_bound, upper_bound) | df[col].isna()

    return df[mask]


def standardize_data(X):
    """
    Standrd Scaler
    """

    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    return X_scaled

def evaluate_knn_rmse(X, y, max_k=30):
    """
    Evaluation of RMSE to find optimal K
    """
    combined = pd.concat([X, y], axis=1).dropna()
    X_clean = combined.drop(columns=y.name)
    y_clean = combined[y.name]

    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
    scores = []

    for k in range(1, max_k):
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        scores.append(rmse)

    return scores

def knn_impute(df_numeric, n_neighbors):
    """
    Imputes missing values in a numeric DataFrame using K-Nearest Neighbors (KNN) imputation.
    """
    
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(df_numeric)
    return pd.DataFrame(imputed_array, columns=df_numeric.columns, index=df_numeric.index)

def find_best_neighbors(df, col_to_impute):
    """
    Finding optimal K
    """
    
    df_numeric = df.select_dtypes(include=[np.number])
    if col_to_impute not in df_numeric.columns:
        raise ValueError(f"Column '{col_to_impute}' must be numeric to evaluate KNN RMSE.")

    y = df_numeric[col_to_impute]
    X = df_numeric.drop(columns=[col_to_impute])
    X_scaled = standardize_data(X)

    rmse_scores = evaluate_knn_rmse(X_scaled, y)
    return rmse_scores.index(min(rmse_scores)) + 1

def impute_knn_column(df, col_to_impute):
    """
    Impute values on a column based on the optimal K found
    """

    df_numeric = df.select_dtypes(include=[np.number])
    if col_to_impute not in df_numeric.columns:
        raise ValueError(f"Column '{col_to_impute}' must be numeric.")

    optimal_k = find_best_neighbors(df, col_to_impute)

    features = df_numeric.drop(columns=[col_to_impute])
    target = df_numeric[[col_to_impute]]

    features_imputed = knn_impute(features, optimal_k)

    df_combined = features_imputed.copy()
    df_combined[col_to_impute] = target

    df_imputed = knn_impute(df_combined, optimal_k)

    return df_imputed[col_to_impute].tolist()

# MF imputer
imputer_med = SimpleImputer(missing_values=np.nan, strategy='median')

def get_filled_values(df, group_col, value_col):
    """
    Fills missing values in `value_col` using the mean of its group based on `group_col`.
    
    If group is missing or group has no mean, the value remains NaN.
    
    Args:
        df (pd.DataFrame): Input DataFrame (can include non-numeric columns).
        group_col (str): Name of the column to group by (e.g., 'pnns_groups_2').
        value_col (str): Name of the numeric column to fill (e.g., 'proteins_100g').

    Returns:
        list: A list of values with NaNs filled where group means are available.
    """
    if value_col not in df.select_dtypes(include=[np.number]).columns:
        raise ValueError(f"'{value_col}' must be a numeric column.")

    # Compute group means (ignoring NaNs)
    mean_dict = df.groupby(group_col, dropna=True)[value_col].mean().to_dict()

    def fill(row):
        val = row[value_col]
        group = row[group_col]

        if pd.notna(val):
            return val
        if pd.isna(group):
            return np.nan
        return mean_dict.get(group, np.nan)

    return df.apply(fill, axis=1).tolist()

### CODE ###

def pre_traitement_file(FILEPATH : str, SAVEPATH : str) -> pd.DataFrame:
    """ 
    Cette fonction a pour but de pré-traiter le fichier venant d'Open Food Facts
    
    Le nettoyage de ce fichier se fait ainsi :
    -> Sélection des produits distribués en France ;
    -> Sélection des informations des produits utiles ;
    -> Suppression des valeurs aberrantes ;
    -> Imputation statistique des valeurs manquantes des colonnes utiles en respectant autant que possible la distribution initiale :
        - Remplissage en fonction du score nutritionnel
        - Remplissage en fonction de la moyenne du sous-groupe alimentaire
        - Remplissage en fonction de la moyenne du groupe alimentaire
        - Remplissage avec la méthode des K-Voisins avec K optimal 
        - Aussi effectué avec Régression Linéaire car plus fidèle à la distribution d'origine

    params: 
        - FILEPATH : filepath where the file from Open Food Facts est stocké localement ;
        - SAVEPATH : filepath where the daatfrale will be saved ;

    returns: 
        File will be saved at the pointed location of "SAVEPATH".

    """

    # read file
    df = pd.read_csv(FILEPATH, sep='\t', lineterminator='\n')

    # select french
    df_fr = df[df['countries'].str.lower().str.contains('fr', na=False)]

    # remove cols that are less 30 complete
    len_30 = int(len(df_fr) * 3/10)
    cols_not_vides = []

    for i in df_fr.columns:
        if df_fr[i].notna().sum() > len_30:
            cols_not_vides.append(i)

    df_fr = df_fr[cols_not_vides]

    # drop dup
    df_fr = df_fr.drop_duplicates(subset=["code"])

    # select useful cols
    df_fr = df_fr[['code', 'product_name', 'main_category_fr', "pnns_groups_1", "pnns_groups_2", 'nutrition_grade_fr', 'additives_n', 
                "additives_fr", 'nutrition-score-fr_100g','energy_100g', 'saturated-fat_100g', 'sugars_100g', 'salt_100g', 'fiber_100g', 'proteins_100g', 
                "carbohydrates_100g"]]

    # replace unknown to have undertandable values when imputing with pnns_groups_2
    df_fr["pnns_groups_2"] = df_fr["pnns_groups_2"].replace("unknown", np.nan)

    # define num cols to impute
    cols_to_impute = ['additives_n', 'nutrition-score-fr_100g', 'energy_100g',
        'saturated-fat_100g', 'sugars_100g', 'salt_100g', 'fiber_100g',
        'proteins_100g', "carbohydrates_100g"]

    # remove outliers
    df_cleaned = remove_outlier_rows(df_fr, cols_to_impute)
    df_cleaned.isna().sum()
    df_cleaned = df_cleaned.reset_index(drop=True)
    df = df_cleaned.copy()

    ############################
    ##### imputation fiber #####
    ############################

    df['fiber_100g_grp'] = get_filled_values(df, "nutrition_grade_fr", "fiber_100g")
    df['fiber_100g_grp'] = get_filled_values(df, "pnns_groups_2", "fiber_100g_grp")
    df['fiber_100g_grp'] = get_filled_values(df, "pnns_groups_1", "fiber_100g_grp")

    imputed_values = impute_knn_column(df, 'fiber_100g_grp')
    df["fiber_100g_grp"] = imputed_values

    df["fiber_100g"] = df["fiber_100g_grp"]

    ############################
    ##### imputation salt ######
    ############################

    df['salt_100g_grp'] = get_filled_values(df, "nutrition_grade_fr", "salt_100g")
    df['salt_100g_grp'] = get_filled_values(df, "pnns_groups_2", "salt_100g")
    df['salt_100g_grp'] = get_filled_values(df, "pnns_groups_1", "salt_100g")

    imputed_values = impute_knn_column(df, 'salt_100g_grp')
    df["salt_100g_grp"] = imputed_values

    imputed_values = impute_knn_column(df, 'salt_100g_grp')
    df["salt_100g_grp"] = imputed_values

    df["salt_100g"] = df["salt_100g_grp"]

    ############################
    #### imputation proteins ###
    ############################

    df['proteins_100g_grp'] = get_filled_values(df, "nutrition_grade_fr", "proteins_100g")
    df['proteins_100g_grp'] = get_filled_values(df, "pnns_groups_2", "proteins_100g_grp")
    df['proteins_100g_grp'] = get_filled_values(df, "pnns_groups_1", "proteins_100g_grp")

    imputed_values = impute_knn_column(df, 'proteins_100g_grp')
    df["proteins_100g_grp"] = imputed_values

    df["proteins_100g"] = df["proteins_100g_grp"]
  
    ############################
    # imputation carbohydrates #
    ############################

    df['carbohydrates_100g_grp'] = get_filled_values(df, "nutrition_grade_fr", "carbohydrates_100g")
    df['carbohydrates_100g_grp'] = get_filled_values(df, "pnns_groups_2", "carbohydrates_100g_grp")
    df['carbohydrates_100g_grp'] = get_filled_values(df, "pnns_groups_1", "carbohydrates_100g_grp")

    imputed_values = impute_knn_column(df, 'carbohydrates_100g_grp')
    df["carbohydrates_100g_grp"] = imputed_values

    df["carbohydrates_100g"] = df["carbohydrates_100g_grp"]

    ############################
    ##### imputation sugars ####
    ############################

    df['sugars_100g_grp'] = get_filled_values(df, "nutrition_grade_fr", "sugars_100g")
    df['sugars_100g_grp'] = get_filled_values(df, "pnns_groups_2", "sugars_100g_grp")
    df['sugars_100g_grp'] = get_filled_values(df, "pnns_groups_1", "sugars_100g_grp")

    imputed_values = impute_knn_column(df, 'sugars_100g_grp')
    df["sugars_100g_grp"] = imputed_values

    df["sugars_100g"] = df["sugars_100g_grp"]

    ############################
    # imputation saturated-fat #
    ############################

    df['saturated-fat_100g_grp'] = get_filled_values(df, "nutrition_grade_fr", "saturated-fat_100g")
    df['saturated-fat_100g_grp'] = get_filled_values(df, "pnns_groups_2", "saturated-fat_100g_grp")
    df['saturated-fat_100g_grp'] = get_filled_values(df, "pnns_groups_1", "saturated-fat_100g_grp")

    imputed_values = impute_knn_column(df, 'saturated-fat_100g_grp')
    df["saturated-fat_100g_grp"] = imputed_values

    df["saturated-fat_100g"] = df["saturated-fat_100g_grp"]

    ############################
    # imputation nutrition-score
    ############################

    train_data = df[['saturated-fat_100g', 'nutrition-score-fr_100g']].dropna()
    X_train = train_data[['saturated-fat_100g']]
    y_train = train_data['nutrition-score-fr_100g']

    predict_data = df[df['nutrition-score-fr_100g'].isna()]
    predict_data = predict_data[predict_data['saturated-fat_100g'].notna()]
    X_predict = predict_data[['saturated-fat_100g']]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_predict)

    df['nutrition-score-fr_100g_lr'] = df['nutrition-score-fr_100g'].copy()
    df.loc[X_predict.index, 'nutrition-score-fr_100g_lr'] = y_pred

    df["nutrition-score-fr_100g"] = df["nutrition-score-fr_100g_lr"]
    
    ############################
    #### imputation energy #####
    ############################

    df['energy_100g_grp'] = get_filled_values(df, "nutrition_grade_fr", "energy_100g")
    df['energy_100g_grp'] = get_filled_values(df, "pnns_groups_2", "energy_100g_grp")
    df['energy_100g_grp'] = get_filled_values(df, "pnns_groups_1", "energy_100g_grp")

    imputed_values = impute_knn_column(df, 'energy_100g_grp')
    df["energy_100g_grp"] = imputed_values

    df["energy_100g"] = df["energy_100g_grp"]

    ############################
    ## imputation n-additives ##
    ############################

    df['additives_n_grp'] = get_filled_values(df, "nutrition_grade_fr", "additives_n")
    df['additives_n_grp'] = get_filled_values(df, "pnns_groups_2", "additives_n_grp")
    df['additives_n_grp'] = get_filled_values(df, "pnns_groups_1", "additives_n_grp")

    imputed_values = impute_knn_column(df, 'additives_n_grp')
    df["additives_n_grp"] = imputed_values
    df["additives_n_grp"] = df["additives_n_grp"].round(0)

    df["additives_n"] = df["additives_n_grp"]

    # remove used cols
    df.drop(columns=["fiber_100g_grp", "salt_100g_grp", "proteins_100g_grp", "carbohydrates_100g_grp", "sugars_100g_grp",
                     "saturated-fat_100g_grp", "nutrition-score-fr_100g_lr", "energy_100g_grp", "additives_n_grp"], inplace=True)
    
    # saving file
    df.to_csv(SAVEPATH, index=False)

# if __name__ == "__main__":
#   pre_traitement_file(FILEPATH : str, SAVEPATH : str)