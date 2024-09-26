import numpy as np
import sympy as sp
import streamlit as st
import pandas as pd
tab1, tab2, tab3 = st.tabs(["Home", "About", "Contact"])

# Display content in tabs

# LU decomposition function
st.markdown(
    """
    <style>
    .custom-subheader {
        font-size: 20px; /* Set the desired font size */
        font-weight: bold; /* Keep the bold effect like a subheader */
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Define the matrix A
A = np.array([[4, 2, 1],
              [2, 5, 3],
              [1, 3, 6]])

# Function for LU decomposition
def lud(A):
    n = A.shape[0]
    L = np.zeros_like(A)
    U = np.zeros_like(A)

    for i in range(n):
        L[i, i] = 1  # Diagonal of L is 1
        for j in range(i, n):
            U[i, j] = A[i, j] - np.sum(L[i, :i] * U[:i, j])
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - np.sum(L[j, :i] * U[:i, i])) / U[i, i]
    
    return L, U

# Function to perform shifting
def shift(A):
    return np.max(np.abs(A)) + 1

# Function to calculate eigenvalues using UL method
def UL_eigen(A, iters=5000, tol=1e-6):
    m, n = A.shape 
    I = np.identity(n)
    shift_A = shift(A) + 1
    A = A + I * shift_A
    
    D1 = A
    D2 = np.ones_like(A)
    k = 0
    i = 0
    
    while (not np.allclose(np.diagonal(D1), np.diagonal(D2), atol=tol)) and i <= iters:
        L, U = lud(D1)
        D2 = np.matmul(U, L)
        
        if np.allclose(np.diagonal(D1), np.diagonal(D2), atol=tol):
            return np.round(np.diagonal(D2) - shift_A, 3)
        
        D1 = D2
        D2 = np.zeros((m, n))
        i += 1
    st.success("Given matrix is not converging(recomended to upload a matrix file of size 7x5 5rows for matrix and 2rows for b1,b2 )")      
    raise ValueError('The eigenvalues did not converge within 5000 iterations try a new file which will converge')



# Determinant calculation from eigenvalues
def calculate_determinant(eigenvalues):
    determinant = np.prod(eigenvalues)
    return determinant

# Check uniqueness of solution
def check_uniqueness(determinant):
    if determinant != 0:
        return "The system has a unique solution."
    else:
        return "The system does not have a unique solution (may have no solution or infinitely many)."

# Construct characteristic polynomial
def construct_polynomial(eigenvalues):
    x = sp.symbols('x')
    polynomial = 1
    for eigenvalue in eigenvalues:
        polynomial *= (x - eigenvalue)
    polynomial = sp.expand(polynomial)
    expanded_str = str(polynomial)

# Replace '**' with '^'
    formatted_str = expanded_str.replace('**', '^')
    return formatted_str+"= 0"

# Function to calculate condition number from eigenvalues
def condition_number(eigenvalues):
    return np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))

# Manually create a Hilbert matrix of size n
def create_hilbert_matrix(n):
    hilbert_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            hilbert_matrix[i][j] = 1 / (i + j + 1)
    return hilbert_matrix

# Solve system (Ax = b) with LU decomposition
def solve_system(L, U, b):
    n = len(b)
    
    # Forward substitution to solve Ly = b
    y = np.zeros_like(b, dtype=np.float64)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    
    # Backward substitution to solve Ux = y
    x = np.zeros_like(b, dtype=np.float64)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    
    return x
def power_method(A, x0, tol=1e-6, max_iter=1000, cycle_check_length=10):
    
    x = x0 / np.linalg.norm(x0)  # Normalize the initial vector
    t_values = []
    
    for i in range(max_iter):
        x_new = np.dot(A, x)
        t = np.linalg.norm(x_new)  # Use norm for scaling
        x_new = x_new / t
        if np.allclose(x, x_new, atol=tol):
            st.write(f"Converged in {i+1} iterations")
            break
            
        # Check for cyclic behavior
        t_values.append(t)
        if len(t_values) > cycle_check_length:
            t_values.pop(0)
            if len(set(t_values)) < cycle_check_length:
                st.write(f"Cyclic behavior detected in {i+1} iterations")
                t = np.sqrt(t_values[-1] * t_values[-2])
                break
            
        x = x_new
    
    eigenvalue = t
    eigenvector = x
    return eigenvalue, eigenvector

# Inverse Power Method function (newly added)
def inverse_power_method(A, x0, tol=1e-6, max_iter=1000, cycle_check_length=10):
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError as e:
        return f"Failed to compute the inverse of A: {e}"
    
    x = x0 / np.linalg.norm(x0)  # Normalize the initial vector
    t_values = []
    
    for i in range(max_iter):
        x_new = np.dot(A_inv, x)
        t = np.linalg.norm(x_new)  # Use norm for scaling
        x_new = x_new / t
        if np.allclose(x, x_new, atol=tol):
            st.write(f"Converged in {i+1} iterations")
            break
            
        t_values.append(t)
        if len(t_values) > cycle_check_length:
            t_values.pop(0)
            if len(set(t_values)) < cycle_check_length:
                st.write(f"Cyclic behavior detected in {i+1} iterations")
                t = np.sqrt(t_values[-1] * t_values[-2])
                break
        
        x = x_new
        
    eigenvalue = 1 / t
    eigenvector = x
    return eigenvalue, eigenvector

# Gauss-Jordan Inversion function (newly added)
def gauss_jordan_inverse(A):
    n = len(A)
    
    if np.linalg.det(A) == 0:
        raise ValueError("Matrix is singular and non-invertible.")
    
    cond_number = np.linalg.cond(A)
    if cond_number > 1e10:
        raise ValueError("Matrix is ill-conditioned with a high condition number. Inversion might not be reliable.")
    
    augmented_matrix = np.hstack((A, np.eye(n)))
    
    for i in range(n):
        diag_element = augmented_matrix[i, i]
        augmented_matrix[i] = augmented_matrix[i] / diag_element
        
        for j in range(n):
            if i != j:
                row_factor = augmented_matrix[j, i]
                augmented_matrix[j] = augmented_matrix[j] - row_factor * augmented_matrix[i]
    
    A_inv = augmented_matrix[:, n:]
    return A_inv

# Streamlit interface
st.sidebar.title("LU Decomposition and Eigenvalue Analysis")
option = st.sidebar.selectbox("Choose an operation", [
    "🚀 Eigenvalues by LU Method",
    "💻 Determinant and System Uniqueness",
    "🎮 Condition Number Comparison",
    "📱 Polynomial Equation and Eigenvalues",
    "💻 Eigenvalues by Power Method and Jordan Technique",
    "🚀 Solving (Ax=b) with LU Decomposition"
])
with tab1:
 st.title("Welcome to the Home Page")
 st.title("Assignment 1.0")
# File uploader for matrix A (Optional CSV upload)
 uploaded_file = st.sidebar.file_uploader("Upload a CSV file for matrix A(Apply the file as given in the input file 5 rows of 5 columns for matrix A and 2 rows just below the matrix for vector b1 and b2 respectively as in the input file)", type="csv")

 if uploaded_file is not None:
    b_vector1 = None
    b_vector2 = None
    data = pd.read_csv(uploaded_file, header=None)

        # Ensure the file has at least 6 columns for matrix A and vector b
    if data.shape[1] < 5:
            st.error("The uploaded file must have at least 6 columns (5 for matrix A and 1 for vector b).")
    else:
            A = data.iloc[:5, :].values
            b_vector1 = data.iloc[5,:].values
            b_vector2 = data.iloc[6,:].values


            # Display matrix A and vectors b and b2
            col1, col2 ,col3= st.columns(3)

        # Display matrix A in the first column
            with col1:
             st.subheader("Matrix A:")
             st.write(A)

        # Display vector b in the second column
            with col2:
             st.subheader("Vector b1:")
             st.write(b_vector1)
            
            with col3:
                st.subheader("Vector b2:")
                st.write(b_vector2)

 else:
    # Default matrix A for demonstration
    A = np.array([
        [4, 1, -2, 2, 3],
        [1, 2, 0, 1, 1],
        [-2, 0, 3, -2, 2],
        [2, 1, -2, -1, 1],
        [3, 1, 2, 1, 2]
    ])
    b_vector1 = np.array([1,2,3,4,5])
    b_vector2 = np.array([1,1,1,1,1])
    col1, col2 ,col3= st.columns(3)

        # Display matrix A in the first column
    with col1:
             st.subheader("Matrix A:(Default)")
             st.write(A)

        # Display vector b in the second column
    with col2:
             st.subheader("Vector b1:(Default)")
             st.write(b_vector1)
            
    with col3:
                st.subheader("Vector b2:(Default)")
                st.write(b_vector2)


 if option == "🚀 Eigenvalues by LU Method":
    eigenvalues = UL_eigen(A)
    
    st.markdown(
    f"""
    <div style="background-color:#f9f9f9; padding:5px; border-radius:5px;">
        <h2 style="color:#4CAF50; text-align:center;">Eigenvalues of A</h2>
        <p style="font-size:24px; color:#555;text-align: center;">λ1 = {eigenvalues[0]} , λ2 = {eigenvalues[1]} , λ3 = {eigenvalues[2]} , λ4 = {eigenvalues[3]} , λ5 = {eigenvalues[4]}</p>
    </div>
    """, 
    unsafe_allow_html=True
    )

 elif option == "💻 Determinant and System Uniqueness":
    eigenvalues = UL_eigen(A)
    det = calculate_determinant(eigenvalues)
    uniqueness = check_uniqueness(det)
    st.subheader(f"Determinant: {det}")
    st.success(uniqueness)

 elif option == "🎮 Condition Number Comparison":
    eigenvalues = UL_eigen(A)
    cond_matrix = condition_number(eigenvalues)
    
    n=5
    hilbert_matrix = create_hilbert_matrix(n)
    cond_hilbert = condition_number(UL_eigen(hilbert_matrix))  # Using the LU method for Hilbert matrix
    
    

# Display subheader with reduced font size
    st.markdown('<p class="custom-subheader">Condition Number of A:</p>', unsafe_allow_html=True)
    st.write(cond_matrix)
    st.markdown('<p class="custom-subheader">Condition Number ofthe Hilbert matrix:</p>', unsafe_allow_html=True)
    st.write(cond_hilbert)
    if cond_matrix > cond_hilbert:
        st.success("Matrix A is ill-conditioned.")
    else:
        st.success("Matrix A is well-conditioned.")

 elif option == "📱 Polynomial Equation and Eigenvalues":
    eigenvalues = UL_eigen(A)
    polynomial = construct_polynomial(eigenvalues)
    st.markdown('<p class="custom-subheader">Characteristic Polynomial is: </p>', unsafe_allow_html=True)
    st.latex(rf"{polynomial}")
    

 elif option == "💻 Eigenvalues by Power Method and Jordan Technique":
    x0 = np.array([1,1,1,1,1])
    leigenvalue, eigenvector = power_method(A,x0)
    st.subheader(f"Largest Eigenvalue: {leigenvalue}")
    st.write("Corresponding Eigenvector:", eigenvector)

    result = inverse_power_method(A, b_vector1)
    if isinstance(result, str):
        st.error(result)
    else:
        smallest_eigenvalue, eigenvector = result
        st.subheader(f"Smallest Eigenvalue: {smallest_eigenvalue}")
        st.write("Corresponding Eigenvector:", eigenvector)

    try:
        A_inv = gauss_jordan_inverse(A)
        st.subheader("Inverse of matrix A:")
        st.write(A_inv)
    except ValueError as e:
        st.error(e)

 elif option == "🚀 Solving (Ax=b) with LU Decomposition":
    # Assume vector b as [1, 2, 3, 4, 5] for demonstration
    
    L, U = lud(A)
    x = solve_system(L, U, b_vector1)
    x1 = solve_system(L, U, b_vector2)
    co1,co2,co3,co4 = st.columns(4)
    with co1:
     st.subheader(f"Solution x1 for Ax=b1:")
     st.write(x)
    with co2: 
     st.subheader(f"Solution x2 for Ax=b2:")
     st.write(x1)
with tab2:
    st.title("About Us")
    ourself = """We are, \n
                 CH23BTECH11040-Saragadam Lokesh 
    CH23BTECH11044-Vardan Gupta
    CH23BTECH11034-Rahul Patil
    CH23BTECH11037-Saket Kashyap
    CH23BTECH11031-Nakul Patole
    ES23BTECH11026-Harshwardhan Matker
    \n"""
    st.write(ourself)
with tab3:
    st.title("Contact Us")
    st.write("""email ids: \n
    ch23btech11040@iith.ac.in
    ch23btech11044@iith.ac.in 
    ch23btech11034@iith.ac.in
    ch23btech11037@iith.ac.in
    ch23btech11031@iith.ac.in
    es23btech11026@iith.ac.in
    \n """)    
