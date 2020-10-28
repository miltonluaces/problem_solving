import recordlinkage as rl
import pandas as pd

# Load Bib Data
bib = pd.read_csv("D:/data/csv/RC_Affiliation.csv") 
bib["Affiliation"] = clean(bib["Affiliation"]) 

# Load GRID Data
grid = pd.read_csv("D:/data/csv/RC_Grid.csv") 
grid["name"] = clean(grid["name"]) 

# Index with Full Index
indexer = rl.FullIndex() 
candidate_links = indexer.index(bib, grid)

# SET UP THE COMPARE OBJECT
compare = rl.Compare(candidate_links, bib, grid)

# BUILT-IN COMPARISON METHODS

# Use built-in comparison functions
compare.exact("Affiliation", "name", name="exact")
compare.string("Affiliation", "name", name="string")

# Print description
print(compare.vectors.describe())

# CUSTOM COMPARISON METHODS

# Import custom functions
from labutils import normed_lcss, normed_fuzzy_lcss

# Perform the comparison
compare.compare(normed_lcss, "Affiliation", "name", name="lcss")
compare.compare(normed_fuzzy_lcss, "Affiliation", "name", name="fuzzy_lcss")

# Print new description, including the 99th and 99.99th percentiles
compare.vectors.describe([0.99, 0.9999])

# DEFINING CUSTOM COMPARISON METHODS

def my_compare(s1, s2):
    # This combines the columns you are comparing into a single DataFrame
    concat = pd.concat([s1, s2], axis=1, ignore_index=True) 

    def inner_apply(x):
       # Create a function to be applied to each pair in the DataFrame.
        val1 = x[0]
        val2 = x[1]
        
        # Do something to produce the result of comparing val1 and val2 return the result
        
    return concat.apply(inner_apply, axis=1)
    

def first_token(s1, s2):
    # This combines the columns you are comparing into a single DataFrame
    concat = pd.concat([s1, s2], axis=1, ignore_index=True)

    def apply_first_token(x):
        # Create a function to be applied to each pair in the DataFrame.
        val1 = x[0]
        val2 = x[1]

        # Do something to produce the result of comparing val1 and val2
        tkn1 = val1.split()
        tkn2 = val2.split()

        score = 0
        if tkn1[0] in tkn2:
            score += 0.5

        if tkn2[0] in tkn1:
            score += 0.5

        # return the result
        return score

    return concat.apply(apply_first_token, axis=1)


