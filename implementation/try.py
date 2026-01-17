import numpy as np

# MAIN PART
# ==============================================================================
# for x, y in zip(xs, ys):
#     for t in range(num_thetas):
#         # compute the rhos for the given point for each theta
#         current_rho = x * coss[t] + y * sins[t]
#         # for each rho, compute the closest rho among the rho_values below it
#         # the index corresponding to that rho is the one we will increase
#         rho_pos = np.where(current_rho > rho_values)[0][-1]
#         # rho_pos = np.argmin(np.abs(current_rho - rho_values))
#         accumulator[rho_pos, t] += 1
# ==============================================================================


# ZIP
# a = np.array([[-1, -2, 3], [-4, 5, 6], [7, 8, -9]])

# xs, ys = np.where(a > 0)

# print(xs, ys)
# for x, y in zip(xs, ys):
#     print(f"x: {x}, y: {y}")

# Where
# current_rho = 2.5
# a = np.array([-1, 2, 5, -6, 7, 4, 2])

# rho_pos = np.where(current_rho < np.abs(a))

# print("rho pos: ", rho_pos)
# print("[0]", np.where(current_rho < np.abs(a))[0])
# print("[0][-1]", np.where(current_rho < np.abs(a))[0][-1])
# print(current_rho - a)
# print(np.argmin(current_rho - a))


# Index array within array

# a = np.array([i for i in range(1, 21)])
# b = np.array([2, 5, 7])

# print(a)
# print(b)
# print(a[b])


# NP ARRAY STUCK

a = np.array([i for i in range(1, 11)])
b = np.array([i for i in range(-10, 0)])

print(np.vstack([a, b]))
print(np.vstack([a, b]).T)
