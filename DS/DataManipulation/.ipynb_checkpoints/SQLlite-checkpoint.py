import sqlite3

# Create database
db = sqlite3.connect('../../../../data/mydb')

# Create table
cursor = db.cursor()
cursor.execute('CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT, phone TEXT, email TEXT unique, password TEXT)')
db.commit()

# Insert data
cursor = db.cursor()
name1 = 'Andres'; phone1 = '3366858'; email1 = 'user@example.com'; password1 = '12345'
 
name2 = 'John'; phone2 = '5557241'; email2 = 'johndoe@example.com'; password2 = 'abcdef'
 
cursor.execute('INSERT INTO users(name, phone, email, password) VALUES(?,?,?,?)''', (name1,phone1, email1, password1))
cursor.execute('INSERT INTO users(name, phone, email, password) VALUES(?,?,?,?)''', (name2,phone2, email2, password2))
 
db.commit()

cursor.execute('''SELECT name, email, phone FROM users''')
user1 = cursor.fetchone() #retrieve the first row
print(user1[0]) #Print the first column retrieved(user's name)
all_rows = cursor.fetchall()
for row in all_rows:
    # row[0] returns the first column in the query (name), row[1] returns email column.
    print('{0} : {1}, {2}'.format(row[0], row[1], row[2]))