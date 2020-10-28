from pandas import read_sql_query
import pandasql
import numpy
 
# CREATE AN IN-MEMORY SQLITE DB
con = connect(":memory:")
cur = con.cursor()
cur.execute("attach 'my.db' as filedb")
cur.execute("create table df as select * from filedb.hflights")
cur.execute("detach filedb")
 
# IMPORT SQLITE TABLE INTO PANDAS DF
df = read_sql_query("select * from df", con)
 
# WRITE QUERIES
sql01 = "select * from df where DayofWeek = 1 and Dest = 'CVG';"
sql02 = "select DayofWeek, AVG(ArrTime) from df group by DayofWeek;"
sql03 = "select DayofWeek, median(ArrTime) from df group by DayofWeek;"
 
# SELECTION:
t11 = pandasql.sqldf(sql01, globals())
 
# AGGREGATION:
t21 = pandasql.sqldf(sql02, globals())
 
