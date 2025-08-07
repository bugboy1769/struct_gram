import itertools
import pandas as pd

my_list = [1, 2, 3, 4]

# Generate all combinations of 2 elements from the list
# The 'r' parameter in combinations specifies the length of the combinations
unique_tuples = list(itertools.combinations(my_list, 2))

print(unique_tuples)

path = "customer.csv"
df = pd.read_csv(path)
truncated_df = df [:15]
print(f"Original DataFrame: {truncated_df}")

columns = truncated_df.columns
series = truncated_df[columns]
print(f"series: {series}")