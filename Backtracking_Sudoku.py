#!/usr/bin/env python
# coding: utf-8

import numpy as np

Sudoku = np.array([[0, 5, 0, 0, 0, 0, 0, 7, 0],
       [3, 0, 2, 0, 0, 7, 5, 0, 0],
       [1, 0, 0, 5, 0, 0, 0, 0, 6],
       [0, 8, 0, 0, 6, 0, 3, 0, 1],
       [0, 0, 0, 0, 7, 0, 0, 0, 0],
       [2, 0, 3, 0, 4, 0, 0, 6, 0],
       [5, 0, 0, 0, 0, 8, 0, 0, 7],
       [0, 0, 4, 9, 0, 0, 1, 0, 3],
       [0, 2, 0, 0, 0, 0, 0, 4, 0]])

# Function to check zero place in sudoku
def zero_check( Sudoku ) :
    height , width = Sudoku.shape
    for i in range(height) :
        for j in range(width) :
            if Sudoku[i][j] == 0 :
                #print(i,j)
                return (i,j)
    return None

#zero_check( Sudoku ) 

#print(Sudoku[:,0])
#np.any(Sudoku[0]==7)

# Function to backtrack to solve sudoku
def backtracking(Sudoku , expected_digit , position) :
    x , y = position
    height , width = Sudoku.shape
    for i in range(height) :
        if Sudoku[x][i] == expected_digit and y != i :
            ##if np.any(Sudoku[x]==expected_digit):
            #print('height - '+str(expected_digit)+' '+str(x)+str(y)+str(i))
            return False
    for i in range(width) :
        if Sudoku[i][y] == expected_digit and x != i :
            ##if np.any(Sudoku[:,y]==expected_digit):
            #print('width - '+str(expected_digit)+' '+str(x)+str(y)+str(i))
            return False
    traverse_x = y//3
    traverse_y = x//3
    for i in range(traverse_y*3,(traverse_y*3)+3) :
        for j in range(traverse_x*3,(traverse_x*3)+3) :
            if Sudoku[i,j]==expected_digit and (i,j) != position:
                #print('height - width - '+str(x)+str(y)+str(i)+str(j))
                return False
    return True

# Function to solve sudoku
def Sudoku_solution( Sudoku ) :
    empty_cell = zero_check( Sudoku )
    if not empty_cell :
        return True
    else :
        x , y = empty_cell
    for digit in range(1,10) :
        if backtracking(Sudoku , digit , (x , y)) :
            Sudoku[x][y] = digit
            #print('Sudoku['+str(x)+']['+str(y)+']'+str(Sudoku[x][y])+' '+str(digit)+' ')
            if Sudoku_solution( Sudoku ):
                #print(Sudoku)
                return True
            Sudoku[x][y] = 0
    return False

Sudoku_solution( Sudoku )
print(Sudoku)
