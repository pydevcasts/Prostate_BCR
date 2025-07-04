# import datetime
# import time 

# start = time.perf_counter()

# class Person:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age
    
#     def show(self):
#         return f"my name is : {self.name} and my age is :{self.age}"
    
#     @classmethod
#     def birth(cls, name, age):
#         return cls(name, datetime.datetime.now().year - age)

# # person = Person("bardiya", 20)
# # person.age = 50
# # print(person.show())

# print(Person.birth("bardiya", 20).age)
# # print(person.age)
# # print(person.name)
# end = time.perf_counter()
# print(end - start)
# ===================================
# def foo(x, y, z=None, *args, **kwargs): paking
#     print(args)
#     print(type(kwargs))
#     print(kwargs)
#     return x + y + args[2]


# print(foo(2, 5, 98989898, 4, 8, 9,655644,545,87, name = "bardiya",age = 20))
# ======================================
# def foo(x, y, z, h):
#     return x

# # kwargs = {"name":"bardiya", "age":20}
# args = (6,8,7,9)
# print(foo(*args))
# =========================================
# from abc import ABC,abstractmethod

# class A(ABC):
#     @abstractmethod
#     def show(self):
#         print(f"I am method of Abstract class")

# class B(A):
#     def show(self):
#         print("heloooooo")

# b = B()
# b.show()
# ======================================
# arr = [1,2,5,4]
# print(dir(arr))
# print(type(arr))
# ==================================
# class Person:
#     def __init__(self, name=None):
#         self.name = name

#     def __show(self):
#         print("heloooooo")


# class Profile(Person):
    
#     def __init__(self,age):
#         self.age = age
#         super().__init__(name="bardiya")


# profile = Profile(25)
# print(profile.age)
# profile._Person__show()
# print(dir(profile))
# =============================================
# class Proile:
#     def __init__(self, name=None):
#         self.name = "bardiya"
    
#     def __call__(self, *args, **kwds):
#         print("hello...")

# p = Proile()
# # print(p.name)
# p.__call__()
# p()
# ================================
# class Proile:
#     def __init__(self, name=None):
#         self.name = "bardiya"
    
#     def __str__(self):
#         return(f"my name is:{self.name}")


# p = Proile()
# print(p.name)
# print(p)
# =================================
# regex
# import re 
# majic_methods
# linked list queue stack
# how to make a decorator
# ===================================
arr = [[1,3,5,7],[2,4,6,8]]
# output = [1,2,3,4,5,6,7,8]