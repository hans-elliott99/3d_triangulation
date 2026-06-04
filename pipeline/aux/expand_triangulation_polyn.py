from sympy import symbols, expand, Poly

t = symbols('t')
f, fp, a, b, c ,d = symbols('f f_prime a b c d')
g = t * (
    (a*t + b)**2 + fp**2 * (c*t + d)**2
    )**2 - (
        (a*d - b*c) * (1 + f**2 * t**2)**2 * (a*t + b) * (c*t + d)
    )
g_expanded = expand(g)
poly = Poly(g_expanded, t)

print("\nThese are the coefficients of the 6 degree polynomial g(t) corresponding to the optimization problem from the reconstruction algorithm")
coefs = poly.all_coeffs()
for i, coef in enumerate(coefs):
    print(f"Degree of t = {6-i}, coef = {coef}")
    # print(f"coefs[{i}] = {coef}")

print("\n\n")
print(poly)

