import regex

# Normal matching.
m1 = regex.search(r'Mr|Mrs', 'Mrs'); print(m1.expandf('{0}'))
m2 = regex.search(r'one(self)?(selfsufficient)?', 'oneselfsufficient'); print(m2.expandf('{0}'))

# POSIX matching.
m3 = regex.search(r'(?p)Mr|Mrs', 'Mrs'); print(m3.expandf('{0}'))
m4 = regex.search(r'(?p)one(self)?(selfsufficient)?', 'oneselfsufficient'); print(m4.expandf('{0}'))

