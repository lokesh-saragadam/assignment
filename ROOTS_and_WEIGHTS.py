import numpy as np
import pandas as pd
import streamlit as st
from sympy import symbols, simplify, init_printing
from scipy.integrate import quad
import matplotlib.pyplot as plt

init_printing()

def jacobi_matrix(n):
    """Creates the Jacobi matrix for the given degree."""
    J = np.zeros((n, n))
    for i in range(1, n):
        J[i, i - 1] = J[i - 1, i] = np.sqrt(i**2 / (4 * i**2 - 1))
    return J

def legendre_polynomial_sympy(n):
    """Generates the nth Legendre polynomial and its coefficients using sympy."""
    x = symbols('x')
    if n == 0:
        return 1, [1]
    elif n == 1:
        return x, [0, 1]

    Pn_minus_2, Pn_minus_1 = 1, x
    for i in range(2, n + 1):
        Pn = simplify(((2 * i - 1) * x * Pn_minus_1 - (i - 1) * Pn_minus_2) / i)
        Pn_minus_2, Pn_minus_1 = Pn_minus_1, Pn

    coefficients = [Pn.coeff(x, i) for i in range(n + 1)]
    return Pn.expand(), coefficients

def companion_matrix_legendre(coefficients):
    """Creates the companion matrix for the polynomial based on its coefficients."""
    n = len(coefficients) - 1
    C = np.zeros((n, n), dtype=float)
    for i in range(1, n):
        C[i, i - 1] = 1
    for j in range(n):
        C[j, n - 1] = -(coefficients[j] / coefficients[n])
    return C

def lagrange_basis(i, x, roots, n):
    """Calculates the ith Lagrange basis polynomial at x."""
    product = 1.0
    for j in range(n):
        if j != i:
            product *= (x - roots[j]) / (roots[i] - roots[j])
    return product

def format_polynomial(coefficients):
    """Formats polynomial coefficients as a LaTeX string."""
    terms = []
    for i, coeff in enumerate(coefficients):
        if coeff != 0:
            term = f"{coeff:.2f}"
            if i == 1:
                term += " x"
            elif i > 1:
                term += f" x^{{{i}}}"
            terms.append(term)
    return " + ".join(terms).replace(" + -", " - ")

# Streamlit Application Interface
st.title('Gauss-Legendre Quadrature Nodes and Weights')

st.sidebar.header("Options")
n = st.sidebar.slider('Select Polynomial Degree (N)', 1, 64, help="Choose the degree of the Legendre polynomial.")
method = st.sidebar.radio(
    'Method to Calculate Nodes and Weights',
    ('Jacobi Matrix', 'Companion Matrix'),
    help="Select the matrix method to find roots and weights."
)

# Display instructions and equations in the main section
st.markdown("""
    This tool calculates the roots and weights for Gauss-Legendre quadrature using different matrix methods.
    Choose the polynomial degree and computation method in the sidebar.
""")

if method == 'Jacobi Matrix':
    with st.spinner("Calculating roots and weights using the Jacobi Matrix..."):
        J = jacobi_matrix(n)
        eigenvalues, eigenvectors = np.linalg.eig(J)
        sorted_indices = np.argsort(eigenvalues)
        roots = eigenvalues[sorted_indices]
        weights = 2 * (eigenvectors[0, sorted_indices] ** 2)

    st.subheader(f'Roots and Weights for Degree {n} (Jacobi Matrix)')
    data = pd.DataFrame({"Roots": roots, "Weights": weights})
    st.dataframe(data.style.format({"Roots": "{:.6f}", "Weights": "{:.6f}"}))

else:
    with st.spinner("Calculating roots and weights using the Companion Matrix..."):
        Pn, coefficients = legendre_polynomial_sympy(n)
        formatted_poly = format_polynomial(coefficients)
        st.sidebar.subheader(f'Legendre Polynomial $P_{{{n}}}(x)$')
        st.sidebar.latex(f"P_{{{n}}}(x) = {formatted_poly}")

        C = companion_matrix_legendre(coefficients)
        roots, _ = np.linalg.eig(C.T)
        roots = np.sort(roots)

        weights = []
        for i in range(n):
            integrand = lambda x: lagrange_basis(i, x, roots, n)
            w_i, _ = quad(integrand, -1, 1)
            weights.append(w_i)
    
    st.subheader(f'Roots and Weights for Degree {n} (Companion Matrix)')
    data = pd.DataFrame({"Roots": roots, "Weights": weights})
    st.dataframe(data.style.format({"Roots": "{:.6f}", "Weights": "{:.6f}"}))

# Save Options
st.sidebar.subheader("Download Options")
if st.sidebar.button('Save as CSV'):
    data.to_csv(f"polynomial_data_{method.lower().replace(' ', '_')}.csv", index=False)
    st.sidebar.success("CSV file saved.")

if st.sidebar.button('Save as Excel'):
    data.to_excel(f"polynomial_data_{method.lower().replace(' ', '_')}.xlsx", index=False)
    st.sidebar.success("Excel file saved.")

# Plotting Options
st.sidebar.header("Plot Customization")
plot_color = st.sidebar.color_picker("Select Plot Color", "#0000ff", help="Choose a color for the plot points.")
point_style = st.sidebar.selectbox("Select Point Style", ['o', 'x', '^', 's'], help="Select a marker style for the plot points.")

# Plot if n == 64 with customization
if n == 64:
    st.subheader("Roots and Weights Plot")
    fig, ax = plt.subplots()
    ax.plot(roots, weights, point_style, color=plot_color, label=method)
    ax.set_xlabel('Roots')
    ax.set_ylabel('Weights')
    ax.legend()
    st.pyplot(fig)
