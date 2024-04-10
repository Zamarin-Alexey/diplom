import my_math
import numpy as np

import direction_calculator
import matplotlib.pyplot as plt

# sll1 = direction_calculator.SLL('pel_data/12.txt', my_math.ApproxMode.POLY, 2, 20)
sll2 = direction_calculator.SLL('pel_data/faz.txt', my_math.ApproxMode.FAZ, 2)

xp = np.linspace(-30, 45)
sll1_arr = []
sll2_arr = []
G = []
for x in xp:
    # sll1_val = sll1.get_sll(x)
    sll2_val = sll2.get_sll(x)
    # sll1_arr.append(sll1_val)
    sll2_arr.append(sll2_val)
    # G.append(sll2_val - sll1_val)

arr = np.loadtxt('pel_data/faz.txt', delimiter='\t', skiprows=1)
plt.plot(arr[:, 0], arr[:, 2])
# plt.plot(xp, sll1_arr, label=r'ДНА1')
plt.plot(xp, sll2_arr, label=r'ДНА2')
# plt.plot(xp, G, label=r'ПЕЛ Х-КА')
plt.xlabel('t')
plt.ylabel('E')
plt.title('E1, E11, E2, E22')
plt.legend(fontsize=16)
plt.minorticks_on()
plt.grid(which='major')
plt.grid(which='minor', linestyle=':')
plt.tight_layout()
plt.show()
