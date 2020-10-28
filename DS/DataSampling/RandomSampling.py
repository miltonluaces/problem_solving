import random

random.seed(444)
random.random()

random.randint(0, 10)
random.uniform(20, 30)

items = ['one', 'two', 'three', 'four', 'five']

# Sample with replacement
random.choice(items)
random.choices(items, k=2)
random.choices(items, k=3)

# Sample without replacement
random.sample(items, 4)

random.shuffle(items)