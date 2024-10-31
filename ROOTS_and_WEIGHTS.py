import numpy as np
import pandas as pd
import streamlit as st
from sympy import symbols, simplify, init_printing
from scipy.integrate import quad
import matplotlib.pyplot as plt

init_printing()

def jacobi_matrix(n):
    J = np.zeros((n, n))
    for i in range(1, n):
        J[i, i - 1] = J[i - 1, i] = np.sqrt(i**2 / (4 * i**2 - 1))
    return J

def legendre_polynomial_sympy(n):
    x = symbols('x')
    if n == 0:
        return 1, [1]
    elif n == 1:
        return x, [0, 1]

    Pn_minus_2 = 1
    Pn_minus_1 = x

    for i in range(2, n + 1):
        Pn = simplify(((2 * i - 1) * x * Pn_minus_1 - (i - 1) * Pn_minus_2) / i)
        Pn_minus_2 = Pn_minus_1
        Pn_minus_1 = Pn
        Pn = Pn.expand()

    coefficients = [Pn.coeff(x, i) for i in range(n + 1)]
    return Pn, coefficients

def companion_matrix_legendre(coefficients):
    n = len(coefficients) - 1
    C = np.zeros((n, n), dtype=float)
    for i in range(1, n):
        C[i, i - 1] = 1
    for j in range(n):
        C[j, n - 1] = -(coefficients[j] / coefficients[n])
    return C

def lagrange_basis(i, x, roots, n):
    product = 1.0
    for j in range(n):
        if j != i:
            product *= (x - roots[j]) / (roots[i] - roots[j])
    return product

def format_polynomial(coefficients, n):
    terms = []
    for i, coeff in enumerate(coefficients):
        if coeff != 0:
            if i == 0:
                terms.append(f"{coeff:.2f}")
            elif i == 1:
                terms.append(f"{coeff:.2f} x")
            else:
                terms.append(f"{coeff:.2f} x^{{{i}}}")
    return " + ".join(terms).replace(" + -", " - ")

st.title('Gauss-Legendre Quadrature Nodes and Weights')

n = st.sidebar.slider('Select Degree of Polynomial (N)', 1, 64)

method = st.sidebar.radio('Select Method', ('Jacobi Matrix', 'Companion Matrix'))

if method == 'Jacobi Matrix':
    J = jacobi_matrix(n)
    eigenvalues, eigenvectors = np.linalg.eig(J)
    
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues_sorted = eigenvalues[sorted_indices]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    
    roots = [eigenvalues_sorted[j] if np.abs(eigenvalues_sorted[j]) >= 1e-6 else 0 for j in range(n)]
    weights = 2 * (eigenvectors_sorted[0] ** 2)
    
    st.subheader(f'Roots and Weights of Legendre Polynomial (Degree {n}) using Jacobi Matrix')

    data = {
        "Roots of Polynomial": roots,
        "Weights of Polynomial": weights
    }
    df = pd.DataFrame(data)
    st.dataframe(df.style.format({
        "Roots of Polynomial": "{:.6f}",
        "Weights of Polynomial": "{:.6f}"
    }))

    if st.button('Save as CSV'):
        df.to_csv("polynomial_data_jacobi.csv", index=False)
        st.success("Saved as polynomial_data_jacobi.csv")
    
    if st.button('Save as Excel'):
        df.to_excel("polynomial_data_jacobi.xlsx", index=False)
        st.success("Saved as polynomial_data_jacobi.xlsx")

else:
    Pn, coefficients = legendre_polynomial_sympy(n)
    formatted_poly = format_polynomial(coefficients, n)
    st.subheader(f'Legendre Polynomial P_{{{n}}}(x)')
    st.latex(f"P_{{{n}}}(x) = {formatted_poly}")
    
    companion_matrix_n = companion_matrix_legendre(coefficients)
    companion_matrix_n = np.array(companion_matrix_n)
    if n>=2:
      st.subheader(f'Companion Matrix of the Legendre Polynomial (Degree {n})')  
      st.dataframe(companion_matrix_n)
    roots, _ = np.linalg.eig(companion_matrix_n.T)
    roots = np.sort(roots)

    weights = []
    for i in range(n):
        integrand = lambda x: lagrange_basis(i, x, roots, n)
        w_i, _ = quad(integrand, -1, 1)
        weights.append(w_i)
    
    weights = np.array(weights)
    
    st.subheader(f'Roots and Weights of Legendre Polynomial (Degree {n}) using Companion Matrix')
    
    data = {
        "Roots of Polynomial": roots,
        "Weights of Polynomial": weights
    }
    df = pd.DataFrame(data)
    st.dataframe(df.style.format({
        "Roots of Polynomial": "{:.6f}",
        "Weights of Polynomial": "{:.6f}"
    }))

    if st.button('Save as CSV'):
        df.to_csv("polynomial_data_companion.csv", index=False)
        st.success("Saved as polynomial_data_companion.csv")
    
    if st.button('Save as Excel'):
        df.to_excel("polynomial_data_companion.xlsx", index=False)
        st.success("Saved as polynomial_data_companion.xlsx")
        

if 1:
    fig, ax = plt.subplots()
    if method == 'Jacobi Matrix':
        ax.plot(roots, weights, 'bo', label='Jacobi Matrix')
    else:
        ax.plot(roots, weights, 'ro', label='Companion Matrix')
    ax.set_xlabel('Roots')
    ax.set_ylabel('Weights')
    ax.legend()
    st.pyplot(fig)
if n>=2:    
    print(companion_matrix_n)
