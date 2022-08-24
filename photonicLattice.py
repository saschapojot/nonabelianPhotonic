import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm
from multiprocessing import Pool
from datetime import datetime

omega=0.01*np.pi

T=2*np.pi/omega

dtEst=0.01
#time step
Q=10#int(T/dtEst)

dt=T/Q
#unit cell number
N=100

dk=2*np.pi/N

phi1=np.pi/3
phi3=np.pi/4

def Jb1(t):
    return np.cos(phi1)

def Jb2(t):
    phi2=omega*t
    return np.sin(phi1)*np.cos(phi2)

def Jc(t):
    phi2=omega*t
    return np.sin(phi1)*np.sin(phi2)*np.cos(phi3)

def Jd(t):
    phi2=omega*t
    return np.sin(phi1)*np.sin(phi2)*np.sin(phi3)


def U(nq):
    """

    :param n: index of momentum
    :param q: index of time
    :return: exponential of -idtH(k,t)
    """
    n,q=nq

    kVal=n*dk
    tq=q*dt
    JbVal=Jb1(tq)+np.exp(1j*kVal)*Jb2(tq)
    JcVal=Jc(tq)
    JdVal=Jd(tq)

    HTmp=csc_matrix(
        [[0,JbVal,JcVal,JdVal],
        [np.conj(JbVal),0,0,0],
        [JcVal,0,0,0],
        [JdVal,0,0,0]
   ] )

    UTmp=expm(-1j*dt*HTmp)

    return [n,q,UTmp]


#state vector in momentum space at all time steps
#row q is vector at time step q at each momentum value
z0=np.zeros((Q+1,N),dtype=complex)
z1=np.zeros((Q+1,N),dtype=complex)
z2=np.zeros((Q+1,N),dtype=complex)
z3=np.zeros((Q+1,N),dtype=complex)


procNum=48

inIndicesForU=[[n,q] for n in range(0,N) for q in range(0,Q)]

pool1=Pool(procNum)

tExpStart=datetime.now()
retIndicesAndU=pool1.map(U,inIndicesForU)
tExpEnd=datetime.now()

print("exp time: ",tExpEnd-tExpStart)

#initial state in real space, the wavefunction is
#completely in sublattice C at unit cell 0
psi0q0=np.zeros(N,dtype=complex)
psi1q0=np.zeros(N,dtype=complex)
psi2q0=np.zeros(N,dtype=complex)
psi2q0[0]=1
psi3q0=np.zeros(N,dtype=complex)

z0[0,:]=np.fft.ifft(psi0q0,norm="ortho")
z1[0,:]=np.fft.ifft(psi1q0,norm="ortho")
z2[0,:]=np.fft.ifft(psi2q0,norm="ortho")
z3[0,:]=np.fft.ifft(psi3q0,norm="ortho")



