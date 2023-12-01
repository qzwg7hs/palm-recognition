import pandas as pd
import ast
import math


def test_cosine():
    for i in range(row_count_test):
        name = test.loc[i, 'name']
        texture = test.loc[i, 'texture']
        gradient = test.loc[i, 'gradient']
        area = test.loc[i, 'area']
        glcm = test.loc[i, 'glcm']

        print("Target image:", name, "Matched image:", closest_matching_cosine(texture, gradient, area, glcm))


def cosine_distance(x, y):
    dot_product = sum(a * b for a, b in zip(x, y))
    norm_x = math.sqrt(sum(a ** 2 for a in x))
    norm_y = math.sqrt(sum(b ** 2 for b in y))

    if norm_x != 0 and norm_y != 0:
        cosine_similarity = dot_product / (norm_x * norm_y)
        return 1 - cosine_similarity

    return 1


def closest_matching_cosine(texture1, grad1, area1, glcm1):
    min_distance = float('inf')
    closest_label = None

    for i in range(row_count_data):
        texture2 = data.loc[i, 'texture']
        cos1 = cosine_distance(texture1, texture2)

        grad2 = data.loc[i, 'gradient']
        cos2 = cosine_distance(grad1, grad2)

        area2 = data.loc[i, 'area']
        cos3 = cosine_distance(area1, area2)

        glcm2 = data.loc[i, 'glcm']
        cos4 = cosine_distance(glcm1, glcm2)

        distance = 0.339 * cos1 + 0.56 * cos2 + 0.1 * cos3 + 0.001 * cos4

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

test_cosine()
