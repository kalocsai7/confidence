import numpy as np

genders = [0.9, 0.8, 0.7, 0.2, 0.3, 0.4]
confs = [0.7, 0.8, 0.6, 0.3, 0.4, 0.5]
gender_threshold = 0.5

# For testing the inside mechanics:
gen_m = genders[0:3]
conf_m = confs[0:3]
gen_f = 1 - np.array(genders[3:6])
conf_f = confs[3:6]

print('Multiply gender values with confidence scores for males:')
print(np.multiply(gen_m, conf_m))

print('\nMultiply gender values with confidence scores for females:')
print(np.multiply(gen_f, conf_f))

print('\nNormalized male value:')
print(np.sum(np.multiply(gen_m, conf_m)) / np.sum(conf_m))

print('\nNormalized female value:')
print(np.sum(np.multiply(gen_f, conf_f)) / np.sum(conf_f))

def estimate_gender(genders, confs, gender_threshold):

    if len(genders) != len(confs):
        print('Gender attribute values and confidence scores should have equal length!')
        return -1

    m = [True if g > gender_threshold else False for g in genders]
    not_m = np.invert(m)

    m_val = np.array(genders)[m]
    m_conf = np.array(confs)[m]
    f_val = np.array(genders)[not_m]
    f_conf = np.array(confs)[not_m]

    male_score = 0 if np.sum(m_conf) == 0 else np.sum(np.multiply(m_val, m_conf)) / np.sum(m_conf)
    female_score = 0 if np.sum(f_conf) == 0 else np.sum(np.multiply(1-f_val, f_conf)) / np.sum(f_conf)

    if male_score == female_score == 0:
        return -1
    return male_score if male_score > female_score else 1 - female_score

print('\nOutput of the "estimate_gender" method:')
print(estimate_gender(genders, confs, gender_threshold))