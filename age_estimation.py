import numpy as np

ages = [30.3, 40.1, 35.3, 37.2, 33.6, 35.7]
confs = [0.4, 0.0, 0.6, 0.8, 0.5, 0.7]

print('Multiply age values with confidence scores:')
print(np.multiply(ages, confs))

print('\nSumming up all instances:')
print(np.sum(np.multiply(ages, confs)))

print('\nNormalizing the sum of instances by the sum of confidence scores:')
print(np.sum(np.multiply(ages, confs)) / np.sum(confs))

def estimate_age(ages, confs):

    if len(ages) != len(confs):
        print('Age attribute values and confidence scores should have equal length!')
        return 0  
        
    return 0 if np.sum(confs) == 0 else np.sum(np.multiply(ages, confs)) / np.sum(confs)

print('\nOutput of the "estimate_age" method:')
print(estimate_age(ages, confs))