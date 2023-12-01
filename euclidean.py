import pandas as pd
import ast


def test_euclidean():
    for i in range(row_count_test):
        name = test.loc[i, 'name']
        texture = test.loc[i, 'texture']
        gradient = test.loc[i, 'gradient']
        area = test.loc[i, 'area']
        glcm = test.loc[i, 'glcm']

        print("Target image: ", name, "Matched image:", closest_matching_euclidean(texture, gradient, area, glcm))


def euclidean_distance(x, y):
    return sum((a - b) ** 2 for a, b in zip(x, y)) ** 0.5


def closest_matching_euclidean(texture1, grad1, area1, glcm1):
    min_distance = float('inf')
    closest_label = None

    for i in range(row_count_data):
        texture2 = data.loc[i, 'texture']
        eucl1 = euclidean_distance(texture1, texture2)

        grad2 = data.loc[i, 'gradient']
        eucl2 = euclidean_distance(grad1, grad2)

        area2 = data.loc[i, 'area']
        eucl3 = euclidean_distance(area1, area2)

        glcm2 = data.loc[i, 'glcm']
        eucl4 = euclidean_distance(glcm1, glcm2)

        distance = 0.49925 * eucl1 + 0.49925 * eucl2 + 0.001 * eucl3 + 0.0005 * eucl4

        if distance < min_distance:
            min_distance = distance
            closest_label = data.loc[i, 'name']

    return closest_label


data = pd.read_csv("data.csv")
test = pd.read_csv("test.csv")

row_count_data = len(data)
row_count_test = len(test)

# string to list conversion
data['texture'] = data['texture'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
data['gradient'] = data['gradient'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
data['area'] = data['area'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
data['glcm'] = data['glcm'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)

test['texture'] = test['texture'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
test['gradient'] = test['gradient'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
test['area'] = test['area'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
test['glcm'] = test['glcm'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)

test_euclidean()
