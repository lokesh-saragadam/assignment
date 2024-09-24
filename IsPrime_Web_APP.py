# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:01:40 2024

@author: LOKESH
"""
import streamlit as st

selected = st.sidebar.selectbox(
    "Main Menu", 
    ["assignment 0.1", "assignment 1.0"]
)
tab1, tab2, tab3 = st.tabs(["Home", "About", "Contact"])

# Display content in tabs
with tab1:
 st.title("Welcome to the Home Page")
   
 if (selected == 'assignment 0.1'):
   st.title('Assignment 0.1')
   def is_prime(n):
     if n <= 1:
        return f"{n} is not a prime number."
     for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return f"{n} is not a prime number."
     return f"{n} is a prime number." 

   def main():
     st.title("Check if a number is a prime number")
       
     number = st.number_input("enter the number here(only integers pls)")
     answer = ''
     if st.button("check"):
         
        if(number >= 2**31):
           print("range exceeded")
        answer  = is_prime(int(number)) 
     st.success(answer)  

   if __name__ == '__main__' :
    main() 
    
if (selected == 'assignment 1.0'):
   st.title('Assignment 1.0')
      
with tab2:
    st.title("About Us")
    ourself = """We are, \n
                 CH23BTECH11040-Saragadam Lokesh 
    CH23BTECH11044-Vardan Gupta
    CH23BTECH11034-Rahul Patil
    CH23BTECH11037-Saket Kashyap
    CH23BTECH11031-Nakul patole
    ES23BTECH11026-Harsh
    \n"""
    st.write(ourself)
with tab3:
    st.title("Contact Us")
    st.write("""email ids: \n
    ch23btech11040@iit.ac.in
    ch23btech11044@iith.ac.in 
    ch23btech11034@iith.ac.in
    ch23btech11037@iith.ac.in
    ch23btech11031@iith.ac.in
    es23btech11026@iith.ac.in
    \n """)     



    

                
   
        
