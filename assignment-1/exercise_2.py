from numpy import linalg as ln
import numpy as np
import random

def randomMatrixGenerator(a,b):
  """Randomly generate AxB matrix from the integers between 0-10"""
  mat = np.random.randint(low=1, high=9, size=(a,b))
  return mat

def findInverseTransMatrix(a,b):
    """Function that does the following tasks:
    1.Randomly generate axb matrix
    2.Transpose the matrix
    3.Find the inverse of the transposed matrix"""  
    mat = randomMatrixGenerator(a,b)
    print("Original matrix is ", mat,"\n")
    tmat = mat.transpose()
    print("Transpose of the original matrix is ", tmat,"\n")
    itmat = np.linalg.inv(tmat)
    print("Inverse of the transposed matrix is ", itmat,"\n")
    return itmat
  
def computeEigens(a,b):
  """Function that compute eigenvalues and eigenvectors of a randomly generated matrix"""
  mat = randomMatrixGenerator(a,b)
  evalue,evector = ln.eig(mat)
  return evalue,evector

#Exercise 2, question 1
random.seed(18)
findInverseTransMatrix(4,4)

#Exercise 2, question 2
random.seed(21)
computeEigens(4,4)
