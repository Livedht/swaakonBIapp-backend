import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('data/course_db.sqlite')
cursor = conn.cursor()

# Query the table schema
cursor.execute('PRAGMA table_info(Kombifildb)')
columns = cursor.fetchall()

# Print column names
for column in columns:
    print(column[1])

conn.close()
