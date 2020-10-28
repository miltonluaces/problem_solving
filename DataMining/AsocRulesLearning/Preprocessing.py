import pandas as pd
from recordlinkage.standardise import clean
from recordlinkage.standardise import phonenumbers


df = pd.read_csv("D:/data/csv/friends.csv", sep='\t')
print(df)
print([df.columns.values])

# CLEAN

# Default Cleaning
df["nameClean"] = clean(df["name"])
print(df)
# Clean the "occupation" column, but keep brackets
df["occupClean"]= clean(df["occupation"], replace_by_none='[^ \\-\\_\(\)A-Za-z0-9]+',remove_brackets=False)
# Clean the phone_number column with replacement
df["phone_number"]= phonenumbers(df["phone_number"])

# VALUE OCCURENCE

from recordlinkage.standardise import value_occurence

from recordlinkage.standardise import value_occurence
import pandas as pd

df = pd.read_csv("friends.csv")

df["household_size"] = value_occurence(df["address"])

# PHONETIC

from recordlinkage.standardise import phonetic
import pandas as pd

# Read in the data
df = pd.read_csv("friends.csv")

# Clean the name column to remove numbers and strip accents
df["name"]= clean(df["name"], replace_by_none='[^ \\-\\_\(\)A-Za-z]+', strip_accents="unicode")

# Standardize using the nysiis phonetic algorithm
df["phonetic"] = phonetic(df["name"], method="nysiis")


