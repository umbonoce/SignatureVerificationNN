import matplotlib.pyplot as plt
from tabulate import tabulate

exp = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']
eer_hmm = [16.896551724135055, 14.482758620701, 12.068965517237876, 10.689655172421272, 9.999999999996867, 4.827586206891981, 16.896551724135055, 8.965517241379018, 4.137931034480347, 4.482758620698806, 3.4482758620618967, 6.896551724158109, 6.896551724145144, 9.999999999996867, 5.862068965517605]
eer_gmm = [12.068965517233426, 11.37931034482171, 14.482758620683258, 8.620689655175493, 6.206896551721722, 0.6896551724137858, 12.068965517233426, 7.93103448277004, 3.1034482758610253, 4.4827586206901735, 4.482758620689102, 4.482758620693248, 4.827586206898362, 6.206896551721722, 4.827586206896868]
paper_hmm = [13.0, 17.0, 14.0, 12.5, 10.5, 3.0, 13.0, 9.0, 5.0, 4.0, 5.0, 6.0, 6.0, 10.0, 8.5]
paper_gmm = [14.0, 16.5, 12.5, 14.0, 9.5, 3.0, 14.0, 8.5, 4.0, 5.0, 5.0, 6.0, 9.5, 6.25]

rows = [[] * 3] * len(exp)

for i in range(0, len(exp)-1):
    rows[i] = [exp[i], round(paper_gmm[i], 2), round(eer_gmm[i], 2)]

print("GMM RESULTS")
print(tabulate(rows, headers=["Exp.", "EER(%) Ref.", "EER(%) Our"]))

for i in range(0, len(exp)-1):
    rows[i] = [exp[i], round(paper_hmm[i],  2), round(eer_hmm[i], 2)]

print("HMM RESULTS")
print(tabulate(rows, headers=["Exp.", "EER(%) Ref.", "EER(%) Our"]))


plt.plot(exp[0:6], eer_hmm[0:6], color='r', linestyle='dashed', marker='o', label='HMM')
plt.plot(exp[0:6], eer_gmm[0:6], color='g', linestyle='dashed', marker='s', label='GMM')
plt.ylabel('EER (%)')
plt.xlabel('Experiments')
plt.title("Ageing Experiments (a-f)")
plt.legend()
plt.show()

plt.plot(exp[6:15], eer_hmm[6:15], color='r', linestyle='dashed', marker='o', label='HMM')
plt.plot(exp[6:15], eer_gmm[6:15], color='g', linestyle='dashed', marker='s', label='GMM')
plt.ylabel('EER (%)')
plt.xlabel('Experiments')
plt.title("Template Update Experiments (f-o)")
plt.legend()
plt.show()
