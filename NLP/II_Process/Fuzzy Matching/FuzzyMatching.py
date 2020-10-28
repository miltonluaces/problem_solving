from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from fuzzyset import FuzzySet

ratio = fuzz.ratio("this is a test", "this is a test!")
print("Perfect ratio = ', ratio")

# Partial ratio (partially matching)
ratio = fuzz.partial_ratio("this is a test", "this is a test!")
print("Partial ratio = ', ratio")

# Token sort ratio 
ratio1 = fuzz.ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
ratio2 = fuzz.token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
print("Normal ratio = ', ratio1")
print("Sort ratio = ', ratio2")

# Token set ratio
ratio1 = fuzz.token_sort_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
ratio2 = fuzz.token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
print("Normal ratio = ', ratio1")
print("Set ratio = ', ratio2")

# Extraction
choices = ["Atlanta Falcons", "New York Jets", "New York Giants", "Dallas Cowboys"]
p1 = process.extract("new york jets", choices, limit=2) 
print("Extract : ', p1")
p2 = process.extractOne("cowboys", choices)
print("Extract one : ', p2")

# Fuzzyset
corpus = """It was a murky and stormy night. I was all alone sitting on a crimson chair. I was not completely alone as I had three felines
    It was a murky and tempestuous night. I was all alone sitting on a crimson cathedra. I was not completely alone as I had three felines
    I was all alone sitting on a crimson cathedra. I was not completely alone as I had three felines. It was a murky and tempestuous night.
    It was a dark and stormy night. I was not alone. I was not sitting on a red chair. I had three cats."""
corpus = [line.lstrip() for line in corpus.split("\n")]
fs = FuzzySet(corpus)
query = "It was a dark and stormy night. I was all alone sitting on a red chair. I was not completely alone as I had three cats."
fs.get(query)
# [(0.873015873015873, 'It was a murky and stormy night. I was all alone sitting on a crimson chair. I was not completely alone as I had three felines')]
