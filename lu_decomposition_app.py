import streamlit as st
import numpy as np
import pandas as pd
import scipy.linalg as la

# Function to perform LU decomposition and solve Ax = b
def solve_system(lu, piv, b_vector):
    x = la.lu_solve((lu, piv), b_vector)
    return x

# Sidebar selection
selected = st.sidebar.selectbox("Main Menu", ["Assignment 0.1", "Assignment 1.0"])

# Home tab
if selected == 'Assignment 1.0':
    st.title('LU Decomposition Solver')

    # File uploader for matrix and vectors
    uploaded_file = st.file_uploader("Upload a CSV file with matrix A and vectors b", type="csv")

    if uploaded_file is not None:
        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file, header=None)

        # Extract the matrix A (5x5) and vectors b1 and b2
        a_matrix = data.iloc[:5].values  # First 5 rows for matrix A
        b_vector1 = data.iloc[5].values  # 6th row for b1
        b_vector2 = data.iloc[6].values  # 7th row for b2

        st.write("Matrix A (5x5):")
        st.write(a_matrix)

        # Display b vectors horizontally
        st.markdown("**Input Vector b1 (5 elements):**")
        b_vector1_display = st.text_input("Enter b1 (comma-separated):", value=','.join(map(str, b_vector1)))

        st.markdown("**Input Vector b2 (5 elements):**")
        b_vector2_display = st.text_input("Enter b2 (comma-separated):", value=','.join(map(str, b_vector2)))

        # Update b vectors from inputs
        b_vector1 = np.array([float(num) for num in b_vector1_display.split(',') if num.strip()])
        b_vector2 = np.array([float(num) for num in b_vector2_display.split(',') if num.strip()])

        # Check determinant before LU decomposition
        determinant = np.linalg.det(a_matrix)
        st.write(f"Determinant of A: {determinant}")

        if abs(determinant) < 1e-10:
            st.error("Matrix A is singular. No unique solution exists.")
        else:
            # Button to compute LU decomposition
            if st.button("Compute LU Decomposition"):
                lu, piv = la.lu_factor(a_matrix)
                st.session_state.lu = lu
                st.session_state.piv = piv
                st.success("LU decomposition computed successfully.")

            # Check if LU decomposition has been computed
            if 'lu' in st.session_state and 'piv' in st.session_state:
                # Solve for the first b
                if st.button("Solve for b1"):
                    try:
                        x1 = solve_system(st.session_state.lu, st.session_state.piv, b_vector1)
                        st.success(f"Solution x1: {x1}")
                    except Exception as e:
                        st.error(f"Error solving for b1: {e}")

                # Solve for the second b
                if st.button("Solve for b2"):
                    try:
                        x2 = solve_system(st.session_state.lu, st.session_state.piv, b_vector2)
                        st.success(f"Solution x2: {x2}")
                    except Exception as e:
                        st.error(f"Error solving for b2: {e}")
