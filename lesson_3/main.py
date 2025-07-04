
# 1-# .  
# a.c  ==> abc =!ac
# 2- ^
# ^hello
# 3-$ 
# $world
# 4- *
# ab*c  ==> abjkghjkgjhgbc ===>abc
# 5-+
# ab+c   => abbhgjhc !=abc 
# 6- ?
# ab?c 
# 7- {n}
# a{3}==> aaa
# 8- {n,}
# a{2,} [2:]
# aa,aaa,aaaaa
# 9-{n,m}
# a{2,4} ==>aa,aaa,aaaa !=aaaaa
# 10-[abc] ==> a,b,c [a-c]
# 11-[^abc] bardiya davo
# 12-[^0-9] 
# 13-\d [0-9]
# 14_\D
# 15-\w  [a-zA-Z0-9_]
# 16_ \W  [^a-zA-Z0-9_]
# 17_ \s \s+
# 18_\S
# ========================
import re 


# res = re.search(r'\d+','There are 1238679 apples')
# print(res.group())

# res = re.match(r"There",'There are 1238679 apples')
# print(res.group())

# res = re.findall(r'\d+','There are 123 apples 245 fgf')
# print(res)

# res = re.sub(r'apples','banana','There are 1238679 apples')
# print(res)

# email = "example@gmail.com"
# pattern = r'^[a-z]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'

# if re.match(pattern, email):
#     print("valid email")
# else:
#     print("invalid")

# text = "Call me at 123-456-7890 or 987-645-3210"

# res = re.findall(r'\d{3}-\d{3}-\d{4}',text)
# print(res)

text ="My mother-in-low is a well-known doctor and 123-test is aprojects"
# ut = ["mother-in-low"]

res=re.findall(r"[a-zA-Z0-9-]+-[a-zA-Z0-9-]+",text)
#for i in res:
 #   if res.match(r"[a-zA-Z0-9-]")
print(res)