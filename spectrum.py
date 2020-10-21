import qutip as qt
import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec
import itertools
import time
import scipy.integrate as integrate
from tabulate import tabulate
from scipy.special import eval_laguerre

kappa=0.2
omega=1
offset=3
K=0.03
T=1
E=1
V=E
V_q=V
V_p=V

MaxN=30

NFock=200

Stabilizer_q=qt.displace(NFock,1j*np.sqrt(2*np.pi))
Stabilizer_p=qt.displace(NFock,np.sqrt(2*np.pi))
Z=qt.displace(NFock,1j*np.sqrt(np.pi/2))
X=qt.displace(NFock,np.sqrt(np.pi/2))


a=qt.destroy(NFock)
a_dag=a.dag()

H_free=E/2*(qt.position(NFock)**2+qt.momentum(NFock)**2)
H_K=K*qt.num(NFock)**2
H_Cass=K*(a**4-offset**4).dag()*(a**4-offset**4)#+K*np.abs(offset)**2

H_stab=-V_q*(2*np.sqrt(np.pi)*qt.position(NFock)).cosm()-V_p*(2*np.sqrt(np.pi)*qt.momentum(NFock)).cosm()



def g_0n(D, Num_max, NFock ):
    """
    returns a D-squeezed 0-ancilla
    """

    r=np.log(1/D)
    psi0=qt.squeeze(NFock, r) * qt.basis(NFock,0)
    psi = qt.Qobj()
    for n in np.arange(-Num_max, Num_max+1):
        psi+=np.exp(-2*np.pi*D**2 * n**2) * qt.displace(NFock, n*np.sqrt(2*np.pi)) * psi0
    return psi.unit()

def Plot_W(rho, res, qmax, name_plot, name, smoothing=False, smooth_center=0,smooth_rad=0, smooth_asymmetry=1, showfig=True):

    xvec = np.linspace(-qmax,qmax,res)
    W=qt.wigner(rho,xvec,xvec,'clenshaw')
    Pq=integrate.simps(W,axis=0)
    Pp=integrate.simps(W,axis=1)

    if smoothing:
    ##smoothing the cutoff-error
        for i in range(res):
            for j in range(res):
                if np.sqrt((xvec[i]/smooth_asymmetry-np.real(smooth_center))**2+(smooth_asymmetry*xvec[j]-np.imag(smooth_center))**2)>=smooth_rad:
                    W[i,j]=0

    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[4, 1],
                           height_ratios=[1, 4]
                           )

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])

    cs=ax3.contourf(xvec, xvec, W, 100, cmap = 'RdBu_r')
    ax3.set_xlabel(r'$q/\sqrt{\pi}$', fontsize=15)
    ax3.set_ylabel(r'$p/\sqrt{\pi}$', fontsize=15)

    tick_max=np.floor(qmax/np.sqrt(np.pi))
    tick_num=np.arange(-tick_max,tick_max+1,1)
    ticks=tick_num*np.sqrt(np.pi)
    tick_lab=[int(x) for x in tick_num]
    ax3.set_xticks(ticks)
    ax3.set_xticklabels(tick_lab)
    ax3.set_yticks(ticks)
    ax3.set_yticklabels(tick_lab)

    ax1.plot(xvec,Pq)
    ax1.set_ylabel(r'$P(q)$', fontsize=15)
    ax1.set_xticks([], [])
    ax1.set_yticks([], [])
    ax4.set_xticks([], [])
    ax4.set_yticks([], [])
    ax4.plot(Pp,xvec)
    ax4.set_xlabel(r'$P(p)$', fontsize=15)
    ax2.text(0.5, 0.5, name_plot,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=20, color='black')

    ax2.set_axis_off()
    if showfig==True:
        plt.show()
    else:
        plt.savefig('Wigner_%s.png'%(name))
        plt.clf()
        plt.cla()
    
    
def Disp(n,m):
    return qt.displace(NFock,n*np.sqrt(np.pi/2)+1j*m*np.sqrt(np.pi/2))

def Prob(n,m, N):
    return (2.)**(-4*N)*sp.special.binom(2*N,n+N)*sp.special.binom(2*N,m+N)

def moment(X, N):
    square = itertools.product(range(-N,N+1,1),range(-N,N+1,1))
    moment = sum(Prob(n,m,N)*np.abs(n*np.sqrt(2*np.pi)+1j*m*np.sqrt(2*np.pi))**X for n,m in square)
    
    return moment

def chan_twirl(S,N):
    square=itertools.product(range(-N,N+1,1),range(-N,N+1,1))
    Twirled = sum(Prob(n,m,N)*qt.to_super(Disp(-n,-m))*S*qt.to_super(Disp(n,m)) for n,m in square)
    return Twirled

def state_twirl(N):
    square=itertools.product(range(-N,N+1,1),range(-N,N+1,1))
    Twirl = sum(Prob(n,m,N)*qt.to_super(Disp(n,m)) for n,m in square)
    return Twirl

def state_twirl_pure(operator, N):
    square=itertools.product(range(-N,N+1,1),range(-N,N+1,1))
    Twirled = sum(Prob(n,m,N)*Disp(n,m)*operator*Disp(n,m).dag() for n,m in square)
    return Twirled

def holevophase(state, quadrature):
    """
    returns estimated peakwidth Delta and circular stabilizer mean phi
    as in eq. (1), Terhal B. M., Weigand D. J.,	arXiv:1909.10075
    """
    if quadrature=='q':
        exp_q=qt.expect(Stabilizer_q, state)
        Delta, phi= np.sqrt(np.log(1/np.abs(exp_q))/np.pi), np.angle(exp_q)/(2*np.sqrt(np.pi))
        
    elif quadrature=='p':
        exp_p=qt.expect(Stabilizer_p, state)
        Delta, phi=np.sqrt(np.log(1/np.abs(exp_p))/np.pi), np.angle(exp_p)/(2*np.sqrt(np.pi))
        
    return Delta, phi

def corr_rot(state):
    RANGE=np.linspace(0,np.pi/2,200)
    Z_list=[]

    for ang in RANGE:
        corr=(1j*ang*qt.num(NFock)).expm()
        GKP_corr=corr*state*(corr.dag())
        Z_list+=[np.abs(qt.expect(Z, GKP_corr))]

    opt_index=Z_list.index(max(Z_list))    
    opt_ang=RANGE[opt_index]

    corr=(1j*opt_ang*qt.num(NFock)).expm()
    GKP_corr=corr*state*(corr.dag())
    
    return GKP_corr

def H_2cat(alpha):
    C_p=(qt.coherent(NFock,alpha)+qt.coherent(NFock,-alpha)).unit()
    C_m=(qt.coherent(NFock,alpha)-qt.coherent(NFock,-alpha)).unit()
    
    H=-(qt.ket2dm(C_p)-qt.ket2dm(C_m))
    
    return H

def H_4cat(alpha):
    
    C_p=(qt.coherent(NFock,alpha)+qt.coherent(NFock,-alpha)).unit()
    C_m=(qt.coherent(NFock,alpha)-qt.coherent(NFock,-alpha)).unit()
    C_ip=(qt.coherent(NFock,1j*alpha)+qt.coherent(NFock,-1j*alpha)).unit()
    C_im=(qt.coherent(NFock,1j*alpha)-qt.coherent(NFock,-1j*alpha)).unit()
    
    C_0=(C_p+C_ip).unit()
    C_1=(C_m+-1j*C_im).unit()
    C_2=(C_p-C_ip).unit()
    C_3=(C_m+1j*C_im).unit()
    
    H=-(qt.ket2dm(C_0)+qt.ket2dm(C_2)-qt.ket2dm(C_1)-qt.ket2dm(C_3))
    
    return H
    
def H_approx_Par(phi):
    H=-np.exp(-phi**2/2)*sum(eval_laguerre(n, phi**2)*qt.fock_dm(NFock,n) for n in range(NFock))

    return H

def H_frust(x):
    
    return (-qt.num(NFock)/x).expm()



##########################################


EV=[]
NRANGE=np.arange(1,MaxN+1,1)
Dq=np.zeros((2,len(NRANGE)))
Dp=np.zeros((2,len(NRANGE)))

for n in range(len(NRANGE)):
    N=NRANGE[n]
    H_eff_par=state_twirl_pure(H_approx_Par(np.sqrt(2*np.pi)), N)

    EV_P, ES_P=H_eff_par.eigenstates()
    EV+=[EV_P]
    

    for i in range(2):

        Dq[i,n]=holevophase(qt.ket2dm(ES_P[i].unit()),'q')[0]
        Dp[i,n]=holevophase(qt.ket2dm(ES_P[i].unit()),'p')[0]




EV-=np.max(EV).real

headerEV=["N"]+["EV_%i"%(i) for i in range(len(EV[0]))]
temp=np.concatenate((NRANGE[:,None],EV.real), axis=1)

file=open('spectrum.txt','w')
file.write(tabulate(temp,headerEV,tablefmt="plain"))
file.close()


#save data
X=np.arange(2)

headerDq=["EV"]+["Dq_%i"%(i) for i in range(len(NRANGE))]
temp=np.concatenate([X[:,None],Dq], axis=1)

file=open('Delta_q.txt','w')
file.write(tabulate(temp,headerDq,tablefmt="plain"))
file.close()

headerDp=["EV"]+["Dp_%i"%(i) for i in range(len(NRANGE))]
temp=np.concatenate([X[:,None],Dp], axis=1)

file=open('Delta_p.txt','w')
file.write(tabulate(temp,headerDp,tablefmt="plain"))
file.close()



fig=plt.figure("spectrum")
for i in range(10):
    if i<=1:
        plt.scatter(NRANGE,EV[:,i], c='r', marker='s')
    else:
        plt.scatter(NRANGE,EV[:,i], c='b', marker='s')
    
    plt.xlabel(r"$N$")
    plt.ylabel(r"$E$")
    plt.xticks(NRANGE, ["%i"%(i) for i in NRANGE])
    plt.savefig("spectrum.png")
    plt.clf()
    plt.cla()

colors=['r','b']
markers=['^','o']
fig=plt.figure("Delta_qp")

for i in range(2):
    plt.plot(NRANGE, Dq[i,:],c='r', marker=markers[i], label='$\Delta_q^{%i}$'%(i))
    plt.plot(NRANGE, Dp[i,:],c='b', marker=markers[i], label='$\Delta_p^{%i}$'%(i))
    
    
    plt.xlabel(r"$N$")
    plt.ylabel(r"$\Delta_{q/p}$")
    plt.xticks(NRANGE, ["%i"%(i) for i in NRANGE])
    plt.legend()
    plt.savefig("deltas.png")
    plt.clf()
    plt.cla()
