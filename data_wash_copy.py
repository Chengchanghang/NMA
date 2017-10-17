# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

lis = []

structs_params = np.load('/home/chengch/NMA/data/10W/data/struct_params.npy')
structs_aligned = np.load('/home/chengch/NMA/data/10W/data/structs_aligned.npy')
IR = np.load('/home/chengch/NMA/data/10W/data/IR.npy')
data = structs_params[:,16]
data = abs(data)

#this si making the split_number for all data
struct1 =[]
struct2= []
struct3 = []
struct4 = []
for _, x in enumerate(data):
    lis.append((_,x))  

for _ in lis:
    if 170 < _[1] <= 180:
        struct1.append(_)
    elif 160 < _[1] <= 170:
        struct2.append(_)
    elif 150 < _[1] <= 160:
        struct3.append(_)
    elif 140 < _[1] <= 150:
        struct4.append(_)

#this is split the struct_params to 140-150, 150-160, 160-170, 170-180, 160-180
struct_params1 =[]
struct_params2 = []
struct_params3 = []
struct_params4 = []

for _ in struct1:
    struct_params1.append(structs_params[_[0]])
for _ in struct2:
    struct_params2.append(structs_params[_[0]])
for _ in struct3:
    struct_params3.append(structs_params[_[0]])
for _ in struct4:
    struct_params4.append(structs_params[_[0]]) 

#
np.save('struct_params1.npy',struct_params1)
np.save('struct_params2.npy',struct_params2)
np.save('struct_params3.npy',struct_params3)
np.save('struct_params4.npy',struct_params4)
#160-180
for _ in struct2:
    struct_params1.append(structs_params[_[0]])
np.save('struct_params1_2.npy',struct_params1)

print ('_____________struct_params finished_____________')
#this is split the IR to 140-150, 150-160, 160-170, 170-180, 160-180
IR1 = []
IR2 = [] 
IR3 = []
IR4 = [] 

for _ in struct1:
    IR1.append(IR[_[0]])
for _ in struct2:
    IR2.append(IR[_[0]])
for _ in struct3:
    IR3.append(IR[_[0]])
for _ in struct4:
    IR4.append(IR[_[0]]) 

np.save('IR1.npy',IR1)
np.save('IR2.npy',IR2)
np.save('IR3.npy',IR3)
np.save('IR4.npy',IR4)
#160-180
for _ in struct2:
    IR1.append(IR[_[0]])
np.save('IR1_2.npy',IR1)
print ('_____________IR finished_____________')

struct_aligned1 = []
struct_aligned2 = []
struct_aligned3 = []
struct_aligned4 = []

for _ in struct1:
    struct_aligned1.append(structs_aligned[_[0]])
for _ in struct2:
    struct_aligned2.append(structs_aligned[_[0]])
for _ in struct3:
    struct_aligned3.append(structs_aligned[_[0]])
for _ in struct4:
    struct_aligned4.append(structs_aligned[_[0]])

np.save('struct_aligned1.npy',struct_aligned1)
np.save('struct_aligned2.npy',struct_aligned2)
np.save('struct_aligned3.npy',struct_aligned3)
np.save('struct_aligned4.npy',struct_aligned4)

for _ in struct2:
    struct_aligned1.append(structs_aligned[_[0]])

np.save('struct_aligned1_2.npy',struct_aligned1)
