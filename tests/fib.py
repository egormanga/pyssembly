a, b = 0, 1
for i in range(1000):
	print(a, end=' ', flush=True)
	a, b = b, a + b
print()
