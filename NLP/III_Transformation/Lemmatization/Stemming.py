from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

pst = PorterStemmer()
lst = LancasterStemmer()
sst = SnowballStemmer("english")

st = lst.stem("comming"); print(st)
st = pst.stem("shoppings"); print(st)
st = sst.stem("building"); print(st)
