# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:01:40 2024

@author: LOKESH
"""
import streamlit as st
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu("assignments",
                           ['assignment 0.1',
                            'assignment 1.0'],
                           default_index=0)
    
if (selected == 'assignment 0.1'):
   st.title('Assignment 0.1')
   def main():
     st.title("Check if a number is a prime number")
       
     number = st.text_input("enter the number here")
     answer = ''
     if st.button("check"):
        answer  = is_prime(int(number)) 
     st.success(answer)  

   if __name__ == '__main__' :
    main() 
    
if (selected == 'assignment 1.0'):
   st.title('Assignment 1.0')
      
    

def is_prime(n):
    if n <= 1:
        return f"{n} is not a prime number."
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return f"{n} is not a prime number."
    return f"{n} is a prime number."

    

                
   
        
