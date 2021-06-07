import matplotlib.pyplot as plt
from tabulate import tabulate

exp = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']
eer_hmm = [16.896551724135055, 14.482758620701, 12.068965517237876, 10.689655172421272, 9.999999999996867, 4.827586206891981, 16.896551724135055, 8.965517241379018, 4.137931034480347, 4.482758620698806, 3.4482758620618967, 6.896551724158109, 6.896551724145144, 9.999999999996867, 5.862068965517605]
eer_gmm = [12.068965517233426, 11.37931034482171, 14.482758620683258, 8.620689655175493, 6.206896551721722, 0.6896551724137858, 12.068965517233426, 7.93103448277004, 3.1034482758610253, 4.4827586206901735, 4.482758620689102, 4.482758620693248, 4.827586206898362, 6.206896551721722, 4.827586206896868]
paper_hmm = [13.0, 17.0, 14.0, 12.5, 10.5, 3.0, 13.0, 9.0, 5.0, 4.0, 5.0, 6.0, 6.0, 10.0, 8.5]
paper_gmm = [14.0, 16.5, 12.5, 14.0, 9.5, 3.0, 14.0, 8.5, 4.0, 5.0, 5.0, 5.0, 6.0, 9.5, 6.25]

random_hmm = [8.620689655175099, 5.4926108374416724, 6.699507389160926, 4.950738916260591, 5.689655172412874, 2.832512315273469, 8.620689655175099, 2.6847290640443755, 0.8128078817748978, 0.9359605911348048, 0.9359605911363533, 2.167487684744208, 1.5517241379348468, 5.689655172412874, 1.7980295566549418]
random_gmm = [7.019704433500265, 6.157635467978439, 6.7241379310449645, 4.137931034489598, 2.33990147782828, 0.0, 7.019704433500265, 2.463054187199458, 0.36945812807986467, 1.4285714285744748, 1.182266009856236, 1.2315270935941465, 1.1083743842394655, 2.33990147782828, 1.4778325123220872]

paper_random_hmm = [5.75, 8.0, 5.0, 7.0, 7.75, 1.0, 5.5, 0.75, 0.0, 0.25, 0.25, 0.25, 1.75, 7.5, 7.0]
paper_random_gmm = [4.0, 9.0, 5.0, 7.25, 7.0, 1.0, 4.0, 0.75, 0.75, 0.75, 0.85, 1.0, 1.5, 7.0, 3.5]

rows = [[] * 3] * len(exp)

for i in range(0, len(exp)-1):
    rows[i] = [exp[i], round(paper_hmm[i], 2), round(random_hmm[i], 2)]

print("GMM RESULTS")
print(tabulate(rows, headers=["Exp.", "EER(%) Ref.", "EER(%) Our"]))

for i in range(0, len(exp)-1):
    rows[i] = [exp[i], round(paper_gmm[i],  2), round(random_gmm[i], 2)]

print("HMM RESULTS")
print(tabulate(rows, headers=["Exp.", "EER(%) Ref.", "EER(%) Our"]))


plt.plot(exp[0:6], random_hmm[0:6], color='r', linestyle='dashed', marker='o', label='nostro')
#plt.plot(exp[0:6], random_gmm2[0:6], color='g', linestyle='dashed', marker='s', label='new')
plt.plot(exp[0:6], paper_random_hmm[0:6], color='b', linestyle='dashed', marker='*', label='paper')
plt.grid(b=True)
plt.ylabel('EER (%)')
plt.xlabel('Experiments')
plt.title("Ageing Experiments (a-f)")
plt.legend()
plt.show()

plt.plot(exp[6:15], random_hmm[6:15], color='r', linestyle='dashed', marker='o', label='nostro')
#plt.plot(exp[6:15], random_gmm2[6:15], color='g', linestyle='dashed', marker='s', label='new')
plt.plot(exp[6:15], paper_random_hmm[6:15], color='b', linestyle='dashed', marker='*', label='paper')
plt.grid(b=True)
plt.ylabel('EER (%)')
plt.xlabel('Experiments')
plt.title("Template Update Experiments (f-o)")
plt.legend()
plt.show()
