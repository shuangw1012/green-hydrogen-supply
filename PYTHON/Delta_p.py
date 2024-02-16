import matplotlib.pyplot as plt
import numpy as np

import os

plt.rcParams["font.family"] = "Times New Roman"


def get_f_D(mf_HC_tube, rho, mu, D, tube_roughness=45e-6): # Darcy Weissbach

    V = mf_HC_tube/(np.pi*(D/2.)**2.*rho)
    Re = D*rho*V/mu
    #f_D = (1.8*np.log10(Re)-1.5)**-2.
    S = np.log(Re/(1.816*np.log(1.1*Re/(np.log(1.+1.1*Re)))))
    f_D = (-2.*np.log10(tube_roughness/(3.71*D)+2.18*S/Re))**(-2.) # Brkic using Lambert W-function approximation to solve Colebrook's implicit equationp.
    return f_D

def p_drop(mf_HC_tube, rho, mu, T, D, L, tube_roughness=45e-6):
    f_D = get_f_D(mf_HC_tube, rho, mu, D, tube_roughness)
    V = mf_HC_tube/(np.pi*(D/2.)**2.*rho)
    Dp_hL = f_D*L/D*rho*V**2./2.
    return Dp_hL

def OneD_deltaP(L,D,z1,z2,m_dot):
    
    '''
    L: total length of pipe, m
    D: pipe diameter, m
    z1: Inlet elevation, m
    z2: Outlet elevation, m
    m_dot: mass flowratem kg/s
    '''
    N_elements = 500
    l=L/N_elements
    import CoolProp.CoolProp as CP

    fluid = 'Hydrogen'
    temperature = 273.15+25  # K
    A = np.pi*D**2/4
    Z = np.linspace(z1,z2,N_elements+1)
    P = np.zeros(N_elements+1)
    P[0] = 150*1e5 # Pa
    for i in range(N_elements):
        z1 = Z[i]
        z2 = Z[i+1]
        p1 = P[i]
        rho = CP.PropsSI('D', 'P', p1, 'T', temperature, fluid)
        mu = CP.PropsSI('V', 'P', p1, 'T', temperature, fluid)
        f_D = get_f_D(m_dot, rho, mu, D, tube_roughness=45e-6)
        V = m_dot/(np.pi*(D/2.)**2.*rho)
        Dp_hL = f_D*l/D*rho*V**2./2.
        Dp = Dp_hL+rho*9.8*(z2-z1)
        p2 = p1 - Dp
        if p2<0:
            print ('pressure error')
            p2=P[0]
        P[i+1]=p2
        
        
    
    return (P[0]/1e5-P[i+1]/1e5),P/1e5

if __name__=='__main__':
    m_dot = 2.115
    L = 100e3
    D=0.15
    z1=184.4
    z2=444.5
    
    delta_P = OneD_deltaP(L,D,z1,z2,m_dot)
    print (delta_P)
    '''
    D_group = np.linspace(0.15,0.6,10)
    Delta_P = np.zeros(len(D_group))
    for i in range(len(D_group)):
        delta_P,P = OneD_deltaP(L,D_group[i],z1,z2,m_dot)
        Delta_P[i] = delta_P
    print (D_group,Delta_P)
    
    fig = plt.figure(figsize=(6, 4))
    plt.plot(D_group*1000,Delta_P)
    plt.xlabel('Pipe diameter, mm')
    plt.ylabel('Pressure drop, bar')
    plt.tight_layout()
    plt.savefig(os.getcwd()+'/delta_p.png',dpi=500)
    plt.close(fig)
    '''
    
    m_dot = 2.115
    L = 500e3
    D=0.15
    z1=184.4
    z2=444.5
    
    delta_P,P = OneD_deltaP(L,D,z1,z2,m_dot)
    print (P)
    L = np.linspace(0,L,501)
    fig = plt.figure(figsize=(6, 4))
    plt.plot(L/1000,P,linestyle = '--')
    plt.xlabel('Pipe length, km')
    plt.ylabel('Pressure, bar')
    plt.ylim(0,160)
    plt.xlim(-10,510)
    plt.tight_layout()
    plt.savefig(os.getcwd()+'/delta_p2.png',dpi=500)
    plt.close(fig)
    