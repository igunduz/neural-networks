from exercise_2 import randomMatrixGenerator
import random

#create random matrix 1
random.seed(5)
mat1 = randomMatrixGenerator(4,4)

#create random matrix 2
random.seed(8)
mat2= randomMatrixGenerator(4,4)

#create random matrix 3
random.seed(18)
mat3= randomMatrixGenerator(4,4)

def check(mat1,mat2,mat3):
    res1 = (mat1 * mat2) * mat3
    res2 = mat1 * (mat2 * mat3)
    if res1.any() != res2.any():
        print("(mat1 * mat2) * mat3 is NOT equal to mat1 * (mat2 * mat3)")
    else:
        print("(mat1 * mat2) * mat3 is equal to mat1 * (mat2 * mat3)")

#check if the equation is TRUE
check(mat1,mat2,mat3)
