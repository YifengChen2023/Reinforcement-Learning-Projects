import numpy as np
from matplotlib import pyplot as plt
f = open("average_rewards.txt", "r" )
a = f.readlines()
data = []
for line in a:
    data.append(line[:-1])
X = np.array(data,dtype=float)
fig1 = plt.figure(figsize=(10,5))
plt.plot(X, color="orange")
plt.xlabel("Episode")
plt.ylabel("average_rewards")
plt.title("Episode_Average_Rewards")
plt.savefig("Episode_Average_Rewards.png")
# plt.show(fig1)



f2 = open("max_rewards.txt", "r" )
a = f2.readlines()
data = []
for line in a:
    data.append(line[:-1])
X = np.array(data,dtype=float)
fig2 = plt.figure(figsize=(10,5))
plt.plot(X, color="orange")
plt.xlabel("Episode")
plt.ylabel("max_rewards")
plt.title("Episode_Max_Rewards")
plt.savefig("Episode_Max_Rewards.png")


f3 = open("loss.txt", "r" )
a = f3.readlines()
data = []
for line in a:
    data.append(line[:-1])
X = np.array(data,dtype=float)
fig3 = plt.figure(figsize=(10,5))
plt.plot(X, color="orange")
plt.xlabel("Episode")
plt.ylabel("loss")
plt.title("Episode_Loss")
plt.savefig("Episode_Loss.png")

f4 = open("numactions.txt", "r" )
a = f4.readlines()
data = []
for line in a:
    data.append(line[:-1])
X = np.array(data,dtype=float)
fig4 = plt.figure(figsize=(10,5))
plt.plot(X, color="orange")
plt.xlabel("Episode")
plt.ylabel("numactions")
plt.title("Episode_Numactions")
plt.savefig("Episode_Numactions.png")