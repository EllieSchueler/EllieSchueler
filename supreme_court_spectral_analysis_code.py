"""Supreme Court Spectral Analysis Code
##Set-Up

Prior to set-up, mount drive or some form of local storarge to load data and save plots as necessary.
"""

#Imports
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Any
from PIL import Image
from tqdm.auto import tqdm
from IPython.display import set_matplotlib_formats
import matplotlib

#Configuration
random.seed(42)
np.random.seed(42)
sns.reset_orig()
set_matplotlib_formats('svg', 'pdf')
matplotlib.rcParams['lines.linewidth'] = 2.0

"""##8-1 Split

"""

#Load the data set. Assume that the sheet has already been screened and sorted into desired era for analysis.
df= pd.read_csv('Supreme Court Data.csv')
df = df[df['maj_votes'] == 8]

#Create data matrix where cases are rows and justices are columns
case_justice_matrix = defaultdict(lambda: defaultdict(int))

for index, row in df.iterrows():
    case_name = row['case_id']
    justice_name = row['justice_name']
    vote = row['vote']
    case_justice_matrix[case_name][justice_name] = vote

case_names = sorted(case_justice_matrix.keys())
justice_names = sorted(set().union(*[case_justice_matrix[case].keys() for case in case_names]))
matrix = np.zeros((len(case_names), len(justice_names)), dtype=int)

# Populate the matrix with entires in the vote column of the original dataset
for i, case_name in enumerate(case_names):
    for j, justice_name in enumerate(justice_names):
      try:
        matrix[i, j] = case_justice_matrix[case_name][justice_name]
      except:
        pass # Handle cases where the justice didn't participate

matrix_df = pd.DataFrame(matrix, index=case_names, columns=justice_names)
matrix_df

# Adjust to read 1 for dissents, 0 otherwise
for i, case_name in enumerate(case_names):
    for j, justice_name in enumerate(justice_names):
      try:
        if case_justice_matrix[case_name][justice_name] == 2:
          matrix[i, j] = 1
        else:
          matrix[i, j] = 0
      except:
        pass # Handle cases where the justice didn't participate

#Create the data vector by summing columns
justice_dissents = matrix.sum(axis=0)
datavec = justice_dissents

#Function for 8-1 spectral analysis

def spec_8_1(dissents):
    v = np.array(dissents, dtype=float)
    n = len(v)

    # Step 1: Compute Mean Effect Vector (m0)
    mean_effect = np.mean(v)
    f0 = np.full_like(v, mean_effect)

    # Step 2: First-Order Effect (M1)
    f1 = v - f0

    # Step 3: Define and project the functions f_x onto M1 and normalize
    gx_functions = []

    for i in range(n):
        gx = np.zeros(n)
        gx[i] = 1
        gx_functions.append(gx)

    gx_projections = [gx - np.mean(gx) for gx in gx_functions]

    normalized_gx_projections = [gx_proj / np.linalg.norm(gx_proj) for gx_proj in gx_projections]

    # Step 4: Normalize f1
    normalized_f1 = f1 / np.linalg.norm(f1)

    # Step 5: Compute inner products of normalized f1 and each normalized fx projection
    inner_products_f1 = np.array([np.dot(normalized_f1, gx_proj) for gx_proj in normalized_gx_projections])

    # Step 6: Return the results
    return {
        "original_f": v,
        "m0 (Mean Effect)": f0,
        "M1 (First-Order Effect)": f1,
        "normalized_f1": normalized_f1,
        "Normalized gx projections": normalized_gx_projections,
        "inner_products_f1": inner_products_f1,
    }

# Print results and index for top inner products

result = spec_8_1(datavec)

inner_products_f1_81 = result['inner_products_f1']
top_4_indices_81 = np.argsort(np.abs(inner_products_f1_81))[-4:]
print("\nTop 4 Inner Products (8-1 split) with f1:")
for i in top_4_indices_81:
    print(f"Justice: {justice_names[i]}, Inner Product: {inner_products_f1_81[i]}")

"""## 7-2 Split"""

df= pd.read_csv('Supreme Court Data.csv')
df = df[df['maj_votes'] == 7]

#Create and fill data matrix as above
#Capture pairwise voting information from the matrix and label justice pairs
justice_names = matrix_df.columns.tolist()
num_justices = len(justice_names)
dissents_together = np.zeros((num_justices, num_justices))

for i in range(num_justices):
    for j in range(i + 1, num_justices):  # Avoid redundant pairs
        justice1 = justice_names[i]
        justice2 = justice_names[j]
        dissents_together[i, j] = np.sum((matrix_df[justice1] == 1) & (matrix_df[justice2] == 1))
        dissents_together[j, i] = dissents_together[i,j]

dissents_vector = dissents_together[np.triu_indices(num_justices, k=1)]

# Create justice pairs labels
justice_pairs = []
for i in range(num_justices):
  for j in range(i+1, num_justices):
      justice_pairs.append((justice_names[i], justice_names[j]))

#Spectral analysis function for 7-2 splits
def spec_7_2(dissents_vector, justice_names):
    v = np.array(dissents_vector, dtype=float)
    num_justices = len(justice_names)
    num_pairs = len(v)

    # Step 1: Mean Effect (M0)
    mean_effect = np.mean(v)
    f0 = np.full_like(v, mean_effect)

    # Step 2: First-Order Effect (M1)
    justice_dissents = np.zeros(num_justices)
    k = 0
    for i in range(num_justices):
        for j in range(i + 1, num_justices):
            justice_dissents[i] += v[k]
            justice_dissents[j] += v[k]
            k += 1
    justice_mean_effect = np.mean(justice_dissents)
    f1_justices = justice_dissents - justice_mean_effect

    # Step 3: First-Order Effect (M2) - Justice Pairs
    f1_pairs = v - f0
    f2_pairs = f1_pairs - np.mean(f1_pairs)

    # Step 5: Normalization
    norm_f1_justices = f1_justices / np.linalg.norm(f1_justices) if np.linalg.norm(f1_justices) != 0 else f1_justices
    norm_f2_pairs = f2_pairs / np.linalg.norm(f2_pairs) if np.linalg.norm(f2_pairs) != 0 else f2_pairs

    # Step 6 & 7: Projection and Inner Products
    gx_functions = [np.zeros(num_justices) for _ in range(num_justices)]
    for i in range(num_justices):
        gx_functions[i][i] = 1
    gx_projections = [gx - np.mean(gx) for gx in gx_functions]
    norm_gx_projections = [gx_proj / np.linalg.norm(gx_proj) if np.linalg.norm(gx_proj) != 0 else gx_proj for gx_proj in gx_projections]
    inner_products_f1 = np.array([np.dot(norm_f1_justices, gx_proj) for gx_proj in norm_gx_projections])

    # Step 8 & 9: Projection and Inner Products - Justice Pairs
    hx_functions = [np.zeros(num_pairs) for _ in range(num_pairs)]
    for i in range(num_pairs):
        hx_functions[i][i] = 1
    hx_projections = [hx - np.mean(hx) for hx in hx_functions]
    norm_hx_projections = [hx_proj / np.linalg.norm(hx_proj) if np.linalg.norm(hx_proj) != 0 else hx_proj for hx_proj in hx_projections]
    inner_products_f2 = np.array([np.dot(norm_f2_pairs, hx_proj) for hx_proj in norm_hx_projections])

    return {
        "original_f": v,
        "m0": f0,
        "m1_justices": f1_justices,
        "m1_pairs": f1_pairs,
        "m2_pairs": f2_pairs,
        "inner_products_f1": inner_products_f1,
        "inner_products_f2": inner_products_f2,
        "justice_names": justice_names,
    }

#Print results and index for top inner products
result_7_2 = spec_7_2(datavec, justice_names)

inner_products_f1_72 = result_7_2['inner_products_f1']
inner_products_f2_72 = result_7_2['inner_products_f2']

top_4_indices_f1_72 = np.argsort(np.abs(inner_products_f1_72))[-4:]
top_4_indices_f2_72 = np.argsort(np.abs(inner_products_f2_72))[-4:]

print("\nTop 4 Inner Products (7-2 split) with f1:")
for i in top_4_indices_f1_72:
    print(f"Justice: {justice_names[i]}, Inner Product: {inner_products_f1_72[i]}")

print("\nTop 4 Inner Products (7-2 split) with f2:")
for i in top_4_indices_f2_72:
  print(f"Justice Pair: {justice_pairs[i]}, Inner Product: {inner_products_f2_72[i]}")

"""## 6-3 Split"""

df= pd.read_csv('Supreme Court Data.csv')
df = df[df['maj_votes'] == 6]

#Create and fill data matrix as above
#Create datavector by labeling and capturing dissents of 3-person coalitions

justice_names = matrix_df.columns.tolist()
num_justices = len(justice_names)

num_triplets = int(num_justices * (num_justices - 1) * (num_justices - 2) / 6)
dissents_together_triplets = np.zeros(num_triplets)

# Create justice triplets labels
justice_triplets = list(itertools.combinations(justice_names, 3))

# Calculate dissent counts for triplets
triplet_index = 0
for justice1, justice2, justice3 in justice_triplets:
    dissents_together_triplets[triplet_index] = np.sum(
        (matrix_df[justice1] == 1) & (matrix_df[justice2] == 1) & (matrix_df[justice3] == 1)
    )
    triplet_index += 1

#Spectral analysis function for 6-3 splits
def spec_6_3(dissents_vector, justice_names, matrix_df):

    v = np.array(dissents_vector, dtype=float)
    num_justices = len(justice_names)
    num_triplets = len(v)

    # Step 1: Mean Effect (m0)
    mean_effect = np.mean(v)
    f0 = np.full_like(v, mean_effect)

    # Step 2: First-Order Effect (M1) - Individual Justices
    justice_dissents = matrix_df.sum(axis=0).values
    justice_mean_effect = np.mean(justice_dissents)
    f1_justices = justice_dissents - justice_mean_effect

    # Step 3: Second-Order Effect (M2) - Justice Pairs (similar to spec_7_2)
    justice_pairs = list(itertools.combinations(range(num_justices), 2))
    pair_dissents = np.array([np.sum((matrix_df.iloc[:, i] == 1) & (matrix_df.iloc[:, j] == 1)) for i, j in justice_pairs])
    pair_mean_effect = np.mean(pair_dissents)
    f2_pairs = pair_dissents - pair_mean_effect


    # Step 4: Third-Order Effect (M3) - Justice Triplets
    f1_triplets = v - f0
    f3_triplets = f1_triplets - np.mean(f1_triplets)

    # Step 5: Normalization
    norm_f1_justices = f1_justices / np.linalg.norm(f1_justices) if np.linalg.norm(f1_justices) != 0 else f1_justices
    norm_f2_pairs = f2_pairs / np.linalg.norm(f2_pairs) if np.linalg.norm(f2_pairs) != 0 else f2_pairs
    norm_f3_triplets = f3_triplets / np.linalg.norm(f3_triplets) if np.linalg.norm(f3_triplets) != 0 else f3_triplets


    # Step 6 & 7: Projection and Inner Products - Individual Justices
    gx_functions = [np.zeros(num_justices) for _ in range(num_justices)]
    for i in range(num_justices):
        gx_functions[i][i] = 1
    gx_projections = [gx - np.mean(gx) for gx in gx_functions]
    norm_gx_projections = [gx_proj / np.linalg.norm(gx_proj) if np.linalg.norm(gx_proj) != 0 else gx_proj for gx_proj in gx_projections]
    inner_products_f1 = np.array([np.dot(norm_f1_justices, gx_proj) for gx_proj in norm_gx_projections])


    # Step 8 & 9: Projection and Inner Products - Justice Pairs
    hx_functions = [np.zeros(len(justice_pairs)) for _ in range(len(justice_pairs))]
    for i in range(len(justice_pairs)):
        hx_functions[i][i] = 1
    hx_projections = [hx - np.mean(hx) for hx in hx_functions]
    norm_hx_projections = [hx_proj / np.linalg.norm(hx_proj) if np.linalg.norm(hx_proj) != 0 else hx_proj for hx_proj in hx_projections]
    inner_products_f2 = np.array([np.dot(norm_f2_pairs, hx_proj) for hx_proj in norm_hx_projections])


    # Step 10 & 11: Projection and Inner Products - Justice Triplets
    ix_functions = [np.zeros(num_triplets) for _ in range(num_triplets)]
    for i in range(num_triplets):
        ix_functions[i][i] = 1
    ix_projections = [ix - np.mean(ix) for ix in ix_functions]
    norm_ix_projections = [ix_proj / np.linalg.norm(ix_proj) if np.linalg.norm(ix_proj) != 0 else ix_proj for ix_proj in ix_projections]
    inner_products_f3 = np.array([np.dot(norm_f3_triplets, ix_proj) for ix_proj in norm_ix_projections])

    return {
        "original_f": v,
        "m0": f0,
        "m1_justices": f1_justices,
        "m2_pairs": f2_pairs,
        "m3_triplets": f3_triplets,
        "inner_products_f1": inner_products_f1,
        "inner_products_f2": inner_products_f2,
        "inner_products_f3": inner_products_f3,
        "justice_names": justice_names,
    }

#Print results and index top inner products
result_6_3 = spec_6_3(datavec, justice_names, matrix_df)

inner_products_f1_63 = result_6_3['inner_products_f1']
inner_products_f2_63 = result_6_3['inner_products_f2']
inner_products_f3_63 = result_6_3['inner_products_f3']
top_4_indices_f1_63 = np.argsort(np.abs(inner_products_f1_63))[-4:]
top_4_indices_f2_63 = np.argsort(np.abs(inner_products_f2_63))[-4:]
top_4_indices_f3_63 = np.argsort(np.abs(inner_products_f3_63))[-4:]


print("\nTop 4 Inner Products (6-3 split) with f1 (Justices):")
for i in top_4_indices_f1_63:
    print(f"Justice: {justice_names[i]}, Inner Product: {inner_products_f1_63[i]}")

print("\nTop 4 Inner Products (6-3 split) with f2 (Pairs):")
for i in top_4_indices_f2_63:
    print(f"Justice Pair: {justice_pairs[i]}, Inner Product: {inner_products_f2_63[i]}")

print("\nTop 4 Inner Products (6-3 split) with f3 (Triplets):")
for i in top_4_indices_f3_63:
    print(f"Justice Triplet: {justice_triplets[i]}, Inner Product: {inner_products_f3_63[i]}")

"""##5-4 Split"""

df= pd.read_csv('Supreme Court Data.csv')
df = df[df['maj_votes'] == 5]

#Create and fill data matrix as above
#Create data vector from the matrix summing over unique justice quartets and assigning labels
justice_names = matrix_df.columns.tolist()
num_justices = len(justice_names)

num_quartets = int(num_justices * (num_justices - 1) * (num_justices - 2) * (num_justices - 3) / 24)
dissents_together_quartets = np.zeros(num_quartets)

justice_quartets = list(itertools.combinations(justice_names, 4))

# Calculate dissent counts for quartets
quartet_index = 0
for justice1, justice2, justice3, justice4 in justice_quartets:
    dissents_together_quartets[quartet_index] = np.sum(
        (matrix_df[justice1] == 1) & (matrix_df[justice2] == 1) &
        (matrix_df[justice3] == 1) & (matrix_df[justice4] == 1)
    )
    quartet_index += 1


datavec = dissents_together_quartets

#Spectral analysis function for 5-4 splits
def spec_5_4(dissents_vector, justice_names, matrix_df):

    v = np.array(dissents_vector, dtype=float)
    num_justices = len(justice_names)
    num_quartets = len(v)

    # Step 1: Mean Effect (m0)
    mean_effect = np.mean(v)
    f0 = np.full_like(v, mean_effect)

    # Step 2: First-Order Effect (M1) - Individual Justices
    justice_dissents = matrix_df.sum(axis=0).values
    justice_mean_effect = np.mean(justice_dissents)
    f1_justices = justice_dissents - justice_mean_effect

    # Step 3: Second-Order Effect (M2) - Justice Pairs
    justice_pairs = list(itertools.combinations(range(num_justices), 2))
    pair_dissents = np.array([np.sum((matrix_df.iloc[:, i] == 1) & (matrix_df.iloc[:, j] == 1)) for i, j in justice_pairs])
    pair_mean_effect = np.mean(pair_dissents)
    f2_pairs = pair_dissents - pair_mean_effect

    # Step 4: Third-Order Effect (M3) - Justice Triplets
    justice_triplets = list(itertools.combinations(range(num_justices), 3))
    triplet_dissents = np.array([np.sum((matrix_df.iloc[:, i] == 1) & (matrix_df.iloc[:, j] == 1) & (matrix_df.iloc[:, k] == 1))
                              for i, j, k in justice_triplets])
    triplet_mean_effect = np.mean(triplet_dissents)
    f3_triplets = triplet_dissents - triplet_mean_effect

    # Step 5: Fourth-Order Effect (M4) - Justice Quartets
    f1_quartets = v - f0
    f4_quartets = f1_quartets - np.mean(f1_quartets)

    # Step 7: Normalization
    norm_f1_justices = f1_justices / np.linalg.norm(f1_justices) if np.linalg.norm(f1_justices) != 0 else f1_justices
    norm_f2_pairs = f2_pairs / np.linalg.norm(f2_pairs) if np.linalg.norm(f2_pairs) != 0 else f2_pairs
    norm_f3_triplets = f3_triplets / np.linalg.norm(f3_triplets) if np.linalg.norm(f3_triplets) != 0 else f3_triplets
    norm_f4_quartets = f4_quartets / np.linalg.norm(f4_quartets) if np.linalg.norm(f4_quartets) != 0 else f4_quartets

    # Step 8 & 9: Projection and Inner Products - Individual Justices
    gx_functions = [np.zeros(num_justices) for _ in range(num_justices)]
    for i in range(num_justices):
        gx_functions[i][i] = 1
    gx_projections = [gx - np.mean(gx) for gx in gx_functions]
    norm_gx_projections = [gx_proj / np.linalg.norm(gx_proj) if np.linalg.norm(gx_proj) != 0 else gx_proj for gx_proj in gx_projections]
    inner_products_f1 = np.array([np.dot(norm_f1_justices, gx_proj) for gx_proj in norm_gx_projections])

    # Step 10 & 11: Projection and Inner Products - Justice Pairs
    hx_functions = [np.zeros(len(justice_pairs)) for _ in range(len(justice_pairs))]
    for i in range(len(justice_pairs)):
        hx_functions[i][i] = 1
    hx_projections = [hx - np.mean(hx) for hx in hx_functions]
    norm_hx_projections = [hx_proj / np.linalg.norm(hx_proj) if np.linalg.norm(hx_proj) != 0 else hx_proj for hx_proj in hx_projections]
    inner_products_f2 = np.array([np.dot(norm_f2_pairs, hx_proj) for hx_proj in norm_hx_projections])

    # Step 12 & 13: Projection and Inner Products - Justice Triplets
    ix_functions = [np.zeros(len(justice_triplets)) for _ in range(len(justice_triplets))]
    for i in range(len(justice_triplets)):
        ix_functions[i][i] = 1
    ix_projections = [ix - np.mean(ix) for ix in ix_functions]
    norm_ix_projections = [ix_proj / np.linalg.norm(ix_proj) if np.linalg.norm(ix_proj) != 0 else ix_proj for ix_proj in ix_projections]
    inner_products_f3 = np.array([np.dot(norm_f3_triplets, ix_proj) for ix_proj in norm_ix_projections])

    # Step 14 & 15: Projection and Inner Products - Justice Quartets
    jx_functions = [np.zeros(num_quartets) for _ in range(num_quartets)]
    for i in range(num_quartets):
        jx_functions[i][i] = 1
    jx_projections = [jx - np.mean(jx) for jx in jx_functions]
    norm_jx_projections = [jx_proj / np.linalg.norm(jx_proj) if np.linalg.norm(jx_proj) != 0 else jx_proj for jx_proj in jx_projections]
    inner_products_f4 = np.array([np.dot(norm_f4_quartets, jx_proj) for jx_proj in norm_jx_projections])


    return {
        "original_f": v,
        "m0": f0,
        "m1_justices": f1_justices,
        "m2_pairs": f2_pairs,
        "m3_triplets": f3_triplets,
        "m4_quartets": f4_quartets,
        "inner_products_f1": inner_products_f1,
        "inner_products_f2": inner_products_f2,
        "inner_products_f3": inner_products_f3,
        "inner_products_f4": inner_products_f4,
        "justice_names": justice_names,
    }

#Compute results using the data vector and index

result_5_4 = spec_5_4(datavec, justice_names, matrix_df)

inner_products_f1_54 = result_5_4['inner_products_f1']
inner_products_f2_54 = result_5_4['inner_products_f2']
inner_products_f3_54 = result_5_4['inner_products_f3']
inner_products_f4_54 = result_5_4['inner_products_f4']

top_4_indices_f1_54 = np.argsort(np.abs(inner_products_f1_54))[-4:]
top_4_indices_f2_54 = np.argsort(np.abs(inner_products_f2_54))[-4:]
top_4_indices_f3_54 = np.argsort(np.abs(inner_products_f3_54))[-4:]
top_4_indices_f4_54 = np.argsort(np.abs(inner_products_f4_54))[-4:]

# Print the top 4 inner products (with corrected labels)
print("\nTop 4 Inner Products (5-4 split) with f1 (Justices):")
for i in top_4_indices_f1_54:
    print(f"Justice: {justice_names[i]}, Inner Product: {inner_products_f1_54[i]}")

print("\nTop 4 Inner Products (5-4 split) with f2 (Pairs):")
for i in top_4_indices_f2_54:
    print(f"Justice Pair: {justice_pairs[i]}, Inner Product: {inner_products_f2_54[i]}")

print("\nTop 4 Inner Products (5-4 split) with f3 (Triplets):")
for i in top_4_indices_f3_54:
    print(f"Justice Triplet: {justice_triplets[i]}, Inner Product: {inner_products_f3_54[i]}")

print("\nTop 4 Inner Products (5-4 split) with f4 (Quartets):")
for i in top_4_indices_f4_54:
    print(f"Justice Quartet: {justice_quartets[i]}, Inner Product: {inner_products_f4_54[i]}")

"""##Squared Norm Calculation"""

#Compute squared norms for each subspace at each split
def compute_squared_norms(result_dict):

    squared_norms = {}
    for key, value in result_dict.items():
        if isinstance(value, np.ndarray) and key.startswith("M"):
            squared_norms[key] = np.sum(value**2)
    return squared_norms


#Compute squared norms for each split level and print
squared_norms_6_3 = compute_squared_norms(result_6_3)
squared_norms_5_4 = compute_squared_norms(result_5_4)
squared_norms_7_2 = compute_squared_norms(result_7_2)
squared_norms_8_1 = compute_squared_norms(result_8_1)

print("Squared Norms for 8-1 Split:")
for key, value in squared_norms_8_1.items():
    print(f"{key}: {value}")

print("Squared Norms for 7-2 Split:")
for key, value in squared_norms_7_2.items():
    print(f"{key}: {value}")

print("Squared Norms for 6-3 Split:")
for key, value in squared_norms_6_3.items():
    print(f"{key}: {value}")

print("\nSquared Norms for 5-4 Split:")
for key, value in squared_norms_5_4.items():
    print(f"{key}: {value}")
