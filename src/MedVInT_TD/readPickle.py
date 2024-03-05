import pickle as p

with open('/home/user/KHJ/PMC-VQA/src/MedVInT_TD/before_load.json', 'rb') as w:
    dict1 = p.load(w)
with open('/home/user/KHJ/PMC-VQA/src/MedVInT_TD/after_load.json', 'rb') as w:
    dict2 = p.load(w)

"""
keys = [key for key in dict1.keys() if key not in dict2.keys()]
print(keys)
"""
keys = [key for key in dict1.keys() if dict1[key] is dict2[key]]
print(keys)

"""print(list(dict1.keys())[:5])
print(list(dict1.values())[:5])"""
print(list(dict2.keys())[:5])
print(list(dict2.values())[:5])
