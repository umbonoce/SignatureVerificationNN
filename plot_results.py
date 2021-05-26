import matplotlib.pyplot as plt

exp = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']
eer_hmm = [15.517241379310354, 10.344827586216837, 13.103448275881775, 10.344827586228398, 3.448275862072981, 1.0344827586187375, 15.517241379310354, 6.551724137948141, 3.448275862070545, 4.82758620689628, 3.4482758620744764, 5.172413793109155, 4.137931034492899, 3.448275862072981, 3.448275862065047]
eer_gmm = [12.068965517233426, 11.37931034482171, 14.482758620683258, 8.620689655175493, 6.206896551721722, 0.6896551724137858, 12.068965517233426, 7.93103448277004, 3.1034482758610253, 4.4827586206901735, 4.482758620689102, 4.482758620693248, 4.827586206898362, 6.206896551721722, 4.827586206896868]


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
