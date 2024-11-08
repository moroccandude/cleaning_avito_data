import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Read the CSV file into a DataFrame
df = pd.read_csv('dt.csv')

# Step 1: Remove "DH" and "/Nuit" from the 'prix' column
df['prix'] = df['prix'].str.replace("DH", "", regex=False)
df['prix'] = df['prix'].str.replace("/Nuit", "", regex=False)

# Step 2: Replace "PRIX NON SPÉCIFIÉ" with NaN
df['prix'] = df['prix'].replace("PRIX NON SPÉCIFIÉ", np.nan)

# Step 3: Remove any remaining spaces within the strings
df['prix'] = df['prix'].apply(lambda x: "".join(x.split()) if isinstance(x, str) else x)

# Step 4: Strip leading/trailing spaces (if any)
df['prix'] = df['prix'].str.strip()

# Step 5: Convert 'prix' to numeric, forcing errors to NaN where necessary
df['prix'] = pd.to_numeric(df['prix'], errors='coerce')

# Convert other columns to numeric
df['chambres'] = pd.to_numeric(df['chambres'], errors='coerce')
df['douches'] = pd.to_numeric(df['douches'], errors='coerce')
df['surface'] = pd.to_numeric(df['surface'], errors='coerce')
df['etage'] = pd.to_numeric(df['etage'], errors='coerce')


# Drop unnecessary columns
df.drop(columns=['Salons', 'Type', 'Âge_bien'], inplace=True)
prix_mean =df['prix'].mean()
prix_median =df['prix'].median()
prix_mode =df['prix'].mode()[0] # mode() returns a series, so we take the first mode
prix_std=df['prix'].std()
print(f'\nMean prix: {prix_mean}')
print(f'Median prix: {prix_median}')
print(f'Mode prix: {prix_mode}')
print(f'std prix: {prix_std}')

chambres_mean =df['chambres'].mean()
chambres_median =df['chambres'].median()
chambres_mode =df['chambres'].mode()[0] # mode() returns a series, so we take the first mode
chambres_std=df['chambres'].std()

print(f'\nMean chambres: {chambres_mean}')
print(f'Median chambres: {chambres_median}')
print(f'Mode chambres: {chambres_mode}')
print(f'std chambres: {chambres_std} \n')


douches_mean =df['douches'].mean()
douches_median =df['douches'].median()
douches_mode =df['douches'].mode()[0] # mode() returns a series, so we take the first mode
douches_std=df['douches'].std()

print(f'\nMean douches: {douches_mean}')
print(f'Median douches: {douches_median}')
print(f'Mode douches: {douches_mode}')
print(f'std douches: {douches_std} \n')

surface_mean =df['surface'].mean()
surface_median =df['surface'].median()
surface_mode =df['surface'].mode()[0] # mode() returns a series, so we take the first mode
surface_std=df['surface'].std()

print(f'\nMean surface: {surface_mean}')
print(f'Median surface: {surface_median}')
print(f'Mode surface: {surface_mode}')
print(f'std surface: {surface_std} \n')

etage_mean =df['etage'].mean()
etage_median =df['etage'].median()
etage_mode =df['etage'].mode()[0] # mode() returns a series, so we take the first mode
etage_std=df['etage'].std()

print(f'\nMean etage: {etage_mean}')
print(f'Median etage: {etage_median}')
print(f'Mode etage: {etage_mode}')
print(f'std etage: {etage_std} \n')


# Check for missing values in the 'prix' column
print('Missing values in prix:', df['prix'].isnull().sum())

# Calculate missing percentage for the whole DataFrame
missing_percentage = df.isnull().mean() * 100
print('Missing percentage:\n', missing_percentage)
df['prix'] = df['prix'].fillna(df['prix'].median())
df['chambres'] = df['chambres'].fillna(df['chambres'].mode().iloc[0])
# inplace() is an argument used in many pandas methods (like fillna(), drop(), rename(), etc.) 
# to decide whether to modify the DataFrame in place or return a new modified DataFrame.
df['douches'] = df['douches'].fillna(df['douches'].mode().iloc[0])
df['surface'] = df['surface'].fillna(df['surface'].median())
df['etage'] = df['etage'].fillna(df['etage'].median())

# il y des valuers frequent so on va utiliser mode pour replacer les valeur manquants
# mode return index anf frequent value 0 5 iloc[0] => to take first
print('Missing values in prix after imputation:', df['prix'].isnull().sum())


# detecter les valeurs aberrantes => outliers
# for i in ['chambres','douches','prix','surface','etage'] :
Q1 = df['prix'].quantile(0.25)  # Premier quartile (25%)
Q3 = df['prix'].quantile(0.75)  # Troisième quartile (75%)
IQR = Q3 - Q1 

Lower_Bound=Q1-1.5*IQR
Upper_Bound=Q3+1.5*IQR
print(f"Lower bound = {Lower_Bound} | upper bound={Upper_Bound}")
i = df[(df['prix'] < Lower_Bound) | (df['prix'] > Upper_Bound)]


df['score_Z'] = (df['douches'] - df['douches'].mean()) / df['douches'].std()
print(df[['score_Z','douches','titre']])
# Un score Z proche de 0 signifie que la donnée est proche de la moyenne.
# Un score Z positif =  moyenne< donnée
# Un score Z négatif = moyenne >  donnée

df['score_Z'] = (df['chambres'] - df['chambres'].mean()) / df['chambres'].std()
print(df[['score_Z','chambres','titre']])

# detecter values aberrantes
Q1 = df['surface'].quantile(0.25)  # Premier quartile (25%)
Q3 = df['surface'].quantile(0.75)  # Troisième quartile (75%)
IQR = Q3 - Q1 

Lower_Bound=Q1-1.5*IQR
Upper_Bound=Q3+1.5*IQR
print(f"Lower bound = {Lower_Bound} | upper bound={Upper_Bound}")
measure = df[(df['surface'] < Lower_Bound) | (df['surface'] > Upper_Bound)]



df['score_Z'] = (df['etage'] - df['etage'].mean()) / df['etage'].std()
print(df[['score_Z','etage','titre']])

plt.figure(figsize=(8, 6))
plt.grid('both')
sns.boxplot(x=df['douches'], color="lightblue", width=0.5)


plt.title("Box Plot for Chambres - Detecting Outliers")
plt.xlabel("Number of douches")

# Show the plot
plt.show()

# Affichage du DataFrame avec les scores Z
print(df)

scaler = StandardScaler()

# List of columns to standardize
columns_to_standardize = ['prix', 'chambres', 'douches', 'surface', 'etage']

# Apply standardization
df_standardized = df.copy()
df_standardized[columns_to_standardize] = scaler.fit_transform(df_standardized[columns_to_standardize])

# Check the results after standardization
print("\nStandardized Data (Z-score):")
print(df_standardized[columns_to_standardize].head())

# Step 14: Normalize (Min-Max scaling)
min_max_scaler = MinMaxScaler()

# Apply normalization (scale values between 0 and 1)
df_normalized = df.copy()
df_normalized[columns_to_standardize] = min_max_scaler.fit_transform(df_normalized[columns_to_standardize])

# Check the results after normalization
print("\nNormalized Data (Min-Max Scaling):")
print(df_normalized[columns_to_standardize].head())

# Final DataFrame with transformations applied
print("\nFinal DataFrame after all transformations and scaling:")
print(df_standardized.head())