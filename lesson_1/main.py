# class A:
#     def __init__(self, name):
#         self.name = name

# class B(A):
#     def __init__(self, age):
#         self.age = age
#         super().__init__(name="bardiya")

#     def call(self):
#         return f"my name is: {self.name} and my age is : {self.age}"
# b = B(22)
# print(b.name)
# print(b.age)
# print(b.call())
# ====================================
# class Profile:
#     def __init__(self, name, familly):
#         self.name = name
#         self.familly = familly
#         # self.email = f"{self.name}_{self.familly}@gmail.com"
#     @property
#     def email(self):
#         return(f"{self.name}_{self.familly}@gmail.com")


# p = Profile("bardiya", "davodi")
# p.name = "korush"
# print(p.email)
# ====================================
class Profile:
    def __init__(self, name):
        self.name = name
    
    def show(self):
        return f"my name is : {self.name}"

    @staticmethod
    def is_age(age):
        if age > 18:
            print(f"I am young pepole")
        elif age < 18 and age > 10:
            print(f"I am teenager ")
        else:
            print(f"I am seniur")

profile = Profile("bardiya")
print(profile.show())
profile.is_age(20)
# =========================================
# class Node:
#     def __init__(self, value=0, next=None):
#         self.value = value
#         self.next = next


# class LinkedList:
#     def __init__(self, head=None):
#         self.head = head
    
#     def to_string(self):
#         node = self.head
#         while node is not None:
#             print(node.value, end = " -> ")
#             if type(node.value) == str:
#                 raise TypeError("invalid type of value ...")
#             node = node.next
#             if node is None:
#                 print("null ..")

# node1 = Node(value=1)
# node2 = Node(value=2)
# node3 = Node(value=3)

# node1.next = node2
# node2.next = node3
# ll = LinkedList(head=node1)
# ll.to_string()
# ==============================================
class Queue:
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        return None
    def front(self):
        if not self.is_empty():
            return self.items[0]
        
    def is_empty(self):
        return len(self.items) == 0
    
q = Queue()
q.enqueue(1)
q.enqueue(2)
q.enqueue(3)
print(q.dequeue())
print(q.dequeue())
print(q.dequeue())
print(q.front())
print(q.is_empty())
# ===========================================
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self,item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        return None
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        return None
    
    def is_empty(self):
        return len(self.items) == 0
    
stack =Stack()
stack.push(1)
stack.push(2)
stack.push(3)
stack.push(4)
print(stack.pop())
print(stack.pop())
print(stack.pop())
print(stack.pop())
print(stack.is_empty())
# ==========================================
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [i for i in arr if i < pivot] #logn
    mid = [i for i in arr if i == pivot] #n
    right = [i for i in arr if i > pivot] #logn
    return quick_sort(left) + mid +  quick_sort(right)
print(quick_sort([34,7,23,32,5,62]))