import matplotlib.pyplot as plt

a = [1, 2, 3]
b = [10, 20, 30]

plt.scatter(a, b, marker="x", color='#de49ca', s=100)
plt.xlabel("A")
plt.ylabel("B")
plt.title("A vs B")
plt.grid(alpha=0.2)
plt.show()