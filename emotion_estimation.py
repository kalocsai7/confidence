import numpy as np

emos = [[0.7, 0.5, 0.6, 0.4, 0.5],
        [0.8, 0.3, 0.6, 0.4, 0.4],
        [0.9, 0.5, 0.7, 0.3, 0.5]]

confs = [0.5, 0.4, 0.6]


print('Multiply emotion values with confidence scores:')
print(np.multiply(np.transpose(emos), confs))

print('\nSumming up over all face instances:')
print(np.sum(np.multiply(np.transpose(emos), confs), axis=1))

print('\nNormalizing the sum of instances by the sum of confidence scores:')
print(np.sum(np.multiply(np.transpose(emos), confs), axis=1)/ np.sum(confs))

def estimate_emotion(emos, confs):

    if len(emos) != len(confs):
        print('Emotion attribute values and confidence scores should have equal length!')
        return [0,0,0,0,0]
        
    return [0,0,0,0,0] if np.sum(confs) == 0 else np.sum(np.multiply(np.transpose(emos), confs), axis=1) / np.sum(confs)

print('\nOutput of the "estimate_emotion" method:')
print(estimate_emotion(emos, confs))