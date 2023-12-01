import pandas as pd
import ast


def test_chi():
    for i in range(row_count_test):
        name = test.loc[i, 'name']
        texture = test.loc[i, 'texture']
        gradient = test.loc[i, 'gradient']
        area = test.loc[i, 'area']
        glcm = test.loc[i, 'glcm']

        print("Target image: ", name, "Matched image:", closest_matching_chi(texture, gradient, area, glcm))


def chi_squared_distance(x, y):
    sum_val = 0
    for xi, yi in zip(x, y):
        if xi + yi > 0:
            sum_val += ((xi - yi) ** 2) / (xi + yi)

    return sum_val


def closest_matching_chi(texture1, grad1, area1, glcm1):
    min_distance = float('inf')
    closest_label = None

    for i in range(row_count_data):
        texture2 = data.loc[i, 'texture']
        chi1 = chi_squared_distance(texture1, texture2)

        grad2 = data.loc[i, 'gradient']
        chi2 = chi_squared_distance(grad1, grad2)

        area2 = data.loc[i, 'area']
        chi3 = chi_squared_distance(area1, area2)

        glcm2 = data.loc[i, 'glcm']
        chi4 = chi_squared_distance(glcm1, glcm2)

        distance = 0.408 * chi1 + 0.59 * chi2 + 0.0019 * chi3 + 0.0001 * chi4

        if distance < min_distance:
            min_distance = distance
            closest_label = data.loc[i, 'name']

    return closest_label


data = pd.read_csv("data.csv")
test = pd.read_csv("test.csv")

row_count_data = len(data)
row_count_test = len(test)

test['texture'] = test['texture'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
test['gradient'] = test['gradient'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
test['area'] = test['area'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
test['glcm'] = test['glcm'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)

data['texture'] = data['texture'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
data['gradient'] = data['gradient'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
data['area'] = data['area'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
data['glcm'] = data['glcm'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)

test_chi()
