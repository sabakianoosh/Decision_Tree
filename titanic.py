import pandas 
import numpy as np
import math  
import csv



csv_data = pandas.read_csv("titanic.csv").head(1000)
s = {'male' : 0 , 'female' : 1 , 'S':0 , 'C' : 1 , 'Q' : 2}
tickett = {'PC 17609' : 0 , '24160' : 1, '113781' : 2, '19952' : 3, '112050' : 4, '11769' : 5, '13502' : 6}
cabinn = {'A36' : 0 , 'B5' : 1, 'C22 C26' : 2, 'D7' : 3, 'E12' : 5}
csv_data['sex'] = csv_data['sex'].map(s)
csv_data['embarked'] = csv_data['embarked'].map(s)
csv_data['ticket'] = csv_data['ticket'].map(tickett)
csv_data["cabin"] = csv_data["cabin"].fillna('B5')
csv_data['age'] = csv_data['age'].fillna(18)
csv_data['cabin'] = csv_data['cabin'].map(cabinn)



csvdata = []
for row in csv_data.values:
    csvdata.append(row)

class node:
    def __init__(self,a, childs, examples, parent, survived=None):
        self.a = a
        self.examples = examples
        self.childs = childs
        self.parent = parent
        self.survived = survived

def Entropy(childs):
    Y = 0
    N = 0
    result = 0 
    totalnumber = 0
    for j in range(len(childs)):
        totalnumber += len(childs[j].examples)
    for nd in childs:
        Y = 0
        N = 0
        for i in nd.examples:
            if i[10]==1:
                Y+=1
            else:
                N+=1
        if (N==0 or Y==0):
            result += 0
        else:
            result += (len(nd.examples)/totalnumber)*(Y/(Y+N)) *(math.log2((Y+N)/Y))
    return result
        





def Entropy1(Node):
    Y = 0
    N = 0
    for i in Node.examples:
        if i[10]==1:
            Y+=1
        else:
            N+=1
    if (N==0 or Y==0):
        return 0
    else:
        return (Y/(Y+N)) *(math.log2((Y+N)/Y))

        
                    
            
def gini_index(childs):
    Y = 0
    N = 0
    result = 0 
    totalnumber = 0
    for j in range(len(childs)):
        totalnumber += len(childs[j].examples)
    for nd in childs:
        for i in nd.examples:
            if i[10]==1:
                Y+=1
            else:
                N+=1
        if (N==0 or Y==0):
            result += 0
        else:
            result += (len(nd.examples)/totalnumber)*((Y/(Y+N))*(1-(Y/(Y+N))) + (N/(Y+N))*(1-(N/(Y+N))))
    return result

        
def sex(Node):
    list1=[]
    list2=[]

    for e in Node.examples:
        if e[2] == 1:
            list1.append(e)
        elif e[2] == 0:
            list2.append(e)
    return [node(None,None,list1,Node),node(None,None,list2,Node)]

def pclass(Node):
    list1=[]
    list2=[]
    list3=[]
    for e in Node.examples:
        if(e[0]==1):
            list1.append(e)
        elif(e[0]==2):
            list2.append(e)
        elif(e[0] == 3):
            list3.append(e)
    return [node(None,None,list1,Node),node(None,None,list2,Node),node(None,None,list3,Node)]  

def age(Node):
    list1=[]
    list2=[]
    list3=[]
    list4=[]
    list5=[]
    for e in Node.examples:
        if(0<e[3]<20):
            list1.append(e)
        elif(20 <= e[3] <40):
            list2.append(e)
        elif(40<= e[3]<60):
            list3.append(e)
        elif(60<= e[3]<80):
            list4.append(e)
        elif(e[3]>80):
            list5.append(e)
        
    return [node(None,None,list1,Node)
            ,node(None,None,list2,Node)
            ,node(None,None,list3,Node)
            ,node(None,None,list4,Node)]
            


def fare(Node):
    list1=[]
    list2=[]
    list3=[]
    list4=[]
    list5=[]
    for e in Node.examples:
        if(0<e[7]<120):
            list1.append(e)
        elif(120 <= e[7] <240):
            list2.append(e)
        elif(240<= e[7]<360):
            list3.append(e)
        elif(360<= e[7]<480):
            list4.append(e)
        elif(e[7]>480):
            list5.append(e)
    return [node(None,None,list1,Node),
            node(None,None,list2,Node),
            node(None,None,list3,Node),
            node(None,None,list4,Node),
            node(None,None,list5,Node)]

def cabin(Node):
    list1=[]
    list2=[]
    list3=[]
    list4=[]
    list5=[]
    for e in Node.examples:
        if(e[8] == 0):
            list1.append(e)
        elif(e[8] == 1):
            list2.append(e)
        elif(e[8] == 2):
            list3.append(e)
        elif(e[8] == 3):
            list4.append(e)
        elif(e[8] == 4):
            list5.append(e)
        
    return [node(None,None,list1,Node),
            node(None,None,list2,Node),
            node(None,None,list3,Node),
            node(None,None,list4,Node),
            node(None,None,list5,Node)]





ListOfattributes = [pclass,sex,age, cabin, fare]
Root = node(None,None,csvdata,None)

def whichquestion2ask(Node,ListOfattributes,Impurity):
    entropy = 1
    selectedQuestion = None
    for a in ListOfattributes:
        Node.childs = a(Node)
        e = Impurity(Node.childs)
        if e < entropy:
            entropy = e
            selectedQuestion = a
    Node.childs = selectedQuestion(Node)
    Node.a = selectedQuestion
    return Node


def check(Node):
    y = 0
    n = 0
    if (Node !=None and Node.examples != None):
        for i in Node.examples:
            if (i[10] == 1):
                y+=1
            else:
                n += 1 
        if y>n:
            return True
        elif n>y:
            return False
        else:
            if(Node.parent!=None):
                return check(Node.parent)
    if(Node.parent!=None):
        return check(Node.parent)
    else:
        return False


def makedecisiontree(node,attributes,Impurity):
    if (len(attributes)==0):
        node.survived = check(node)
        return None
    if (len(node.examples ) ==0 ):
        node.survived = check(node.parent)
        return None
    node = whichquestion2ask(node,attributes,Impurity)
    for child in node.childs:
        makedecisiontree(child,list(filter(lambda s: s!= node.a,attributes)),Impurity)
    return node

# Impurity = Entropy
Impurity = gini_index
makedecisiontree(Root,ListOfattributes,Impurity)


def harrass(Node):
    if((Node.childs)!=None):
        for ch in Node.childs:
            harrass(ch)

    else:
       
       if(Node.parent.childs!=None and (Entropy1(Node.parent) - Entropy(Node.parent.childs)) < 0.02):
           Node.parent.childs = None
           Node.parent.survived = check(Node.parent)
           return None
    
        
harrass(Root)


newnode = node(None,None,[csv_data.values[62]],None)


def isAlive(person,Root):
    childs = Root.a(person)
    j = 0
    selectedindex = 0
    for c in childs:
        j+=1
        if (len(c.examples)!=0):
            selectedindex = j
            break
    if(Root.childs!= None):
        if (Root.childs[selectedindex-1].childs == None):
            return Root.childs[selectedindex-1].survived
        else:
            return isAlive(person,Root.childs[selectedindex-1])    
        
isAlive(newnode,Root)



total = len(csv_data)
trues= 0
for i in range (len(csv_data)):
    state = None
    if (isAlive(node(None,None,[csv_data.values[i]],None),Root)==True):
        state = 1
    else:
        state = 0
    if (state==csv_data.values[i][-1]):
        trues+=1
print(trues/total)
