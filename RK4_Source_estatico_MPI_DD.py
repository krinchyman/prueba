# -*- coding: utf-8 -*-

from mpi4py import MPI
from scipy.integrate import odeint
from numpy.linalg import inv
import numpy as np

from netCDF4 import Dataset

from numba import autojit

import argparse

def crea_evolucion(fout, x):

    destino = Dataset(filename=fout, mode='w', clobber=True, format='NETCDF3_CLASSIC')

    destino.createDimension('time', None)
    destino.createDimension('x', Nx-ghost+1)

    variable    = destino.createVariable('x','f4',('x',))
    variable[:] = x[ghost:Nx+1]
    variable.standard_name = "x"
    variable.long_name = "Coordenada holografica"

    variable         = destino.createVariable('time','f4',('time',))
    variable.standard_name = "time"
    variable.long_name = "time"

    variables = ['A', 'delta', 'Pi', 'Phi', 'MomentumConstraint']

    for nombre in variables:

        variable    = destino.createVariable(nombre,'f4',('time','x'))

    destino.close()

def buffer_out(fout, t, Phi, Pi, A, delta, constraint, **kwargs):

    variables = {'A' : A, 'delta' : delta, 'Pi' : Pi, 'Phi' : Phi, 'MomentumConstraint' : constraint}

    destino = Dataset(filename=fout, mode='a')

    Nt = len(destino.dimensions['time'])

    destino.variables['time'][Nt] = t

    for n, v in variables.iteritems():

        variable       = destino.variables[n]
        variable[Nt,:] = v[ghost:Nx+1]

    # Atribulos del run:
    if kwargs:

        destino.setncatts(kwargs)

    destino.close()

    return
    
def crea_series(fout):

    destino = Dataset(filename=fout, mode='w', clobber=True, format='NETCDF3_CLASSIC')

    destino.createDimension('time', None)

    variable               = destino.createVariable('time','f4',('time',))
    variable.standard_name = "time"
    variable.long_name     = "time"

    variables = ['Amin', 'Masa', 'Pi0', 'VeV']

    for nombre in variables:

        variable    = destino.createVariable(nombre,'f4',('time'))
        variable.units = ''

    destino.close()    

def escribe_series(fout, buffer_series):

    destino = Dataset(filename=fout, mode='a')
        
    Nt = len(destino.dimensions['time'])

    destino.variables['time'][Nt:] = np.array(buffer_series['time'])[:]

    for n,v in buffer_series.items():
        
        destino.variables[n][Nt:] = np.array(v)[:]

    destino.close()

def BC(Phi, Pi, t):

    ''' Von Neumann boundary conditions. '''
    
    if rank==0:

        Phi[ghost] = 0.

        X  = dx*np.arange(-ghost,0)
        A0 = np.matrix([[1,0,0],
                        [1, dx, dx**2],
                        [1, 2*dx, 4*dx**2]])

        for f in [Phi, Pi]:

            # Cálculo de los coeficiente:
            res = np.dot(inv(A0),f[ghost:ghost+3])
            [[a0,b0,c0]] = np.matrix.tolist(res)

            # Extrapolación:
            f[0:ghost] = a0 + b0*X + c0*X**2


    if rank==size-1:

        Phi[Nx] = 0.
        Pi[Nx]  = Pib

        X  = dx*np.arange(1,ghost+1)
        Ab = np.matrix([[1,-2*dx, 4*dx**2],
                        [1,-dx, dx**2],
                        [1,0,0]])

        for f in [Phi, Pi]:

            # Cálculo de los coeficiente:
            res = np.dot(inv(Ab),f[Nx-2:Nx+1])
            [[ab,bb,cb]] = np.matrix.tolist(res)

            # Extrapolación:
            f[Nx+1:Nx+ghost+1] = ab + bb*X + cb*X**2

    return

############################################################
# Funciones necesarias para el cálculo de los initial data #
############################################################

def eqs(y, r, params):
    
    # Parámetros:
    alpha, = params
    
    # Punto en el que se encuentra la integración:
    A, delta = y
    
    # Ahora las derivadas:
    f1 = -alpha**2*np.exp(2*delta)*np.sin(r)*np.cos(r)/A + (-A + 1)*(2*np.sin(r)**2 + 1)/(np.sin(r)*np.cos(r))
    f2 = -alpha**2*np.exp(2*delta)*np.sin(r)*np.cos(r)/A**2
    
    return [f1, f2]

def InitialPi_new(alpha, points):
    
    # Los puntos en donde queremos resolver el sistema:
    r      = np.concatenate( [ [10**-8], np.linspace(0.,np.pi/2,points)[1:] ])

    # La condición inicial:
    As     = 1 - alpha**2*r**2/3 + alpha**2*(2 - 3*alpha**2 + 3*(4 + 3*alpha**2))*r**4/45 \
               - r**6*alpha**2*(90*alpha**4 + 9*(42 - 45*alpha**4)  \
               - 6*(-8 + 60*alpha**2 + 45*alpha**4)  \
               + 27*(32 + 75*alpha**2 + 45*alpha**4))/14175

    deltas = alpha**2*r**2/2 +  (alpha**2/6 + alpha**4/12)*r**4 \
             - r**6*alpha**2*(90*alpha**4 - 45*alpha**2*(4 + alpha**2) \
             + 9*(4 - 30*alpha**2 - 45*alpha**4) \
             + 27*(2 + 15*alpha**2 + 15*alpha**4))/4050
    
    # La condición inicial:
    y0     = (As[0],deltas[0])

    # El parámetro que define la pendiente del source:
    params = [alpha,]

    psoln = odeint(eqs, y0, r, args=(params,), full_output=0, printmessg=True, rtol=1E-11, atol=1e-11)

    # Obtenemos nuestras soluciones:
    A     = psoln[:,0]
    delta = psoln[:,1]

    # El primer punto no tiene la solución en 0 si no en 10**-8. Corregimos con los valores conocidos:
    A[0]     = 1.
    delta[0] = 0.

    Pi    = alpha*np.exp(delta)/A
    
    return np.concatenate([np.zeros(ghost), Pi, np.zeros(ghost)])

def InitialPi_nc(fichero):

    datos = Dataset(fichero)

    Pi  = datos.variables['Pi'][-1,:]

    return np.concatenate([np.zeros(ghost), Pi, np.zeros(ghost)])


def InitialPhi_nc(fichero):

    datos = Dataset(fichero)

    Phi  = datos.variables['Phi'][-1,:]

    return np.concatenate([np.zeros(ghost), Phi, np.zeros(ghost)])

############################################################

def sync(vector):

    ''' Esta función sincroniza los buffers de ghost points '''

    # Izquierda a derecha:
    comm.Sendrecv(vector[ghost:2*ghost], dest=ipl, recvbuf=vector[-ghost:], source=ipr)
    comm.Barrier()

    # Derecha a izquierda:
    comm.Sendrecv(vector[Nx-ghost+1:Nx+1], dest=ipr, recvbuf=vector[0:ghost], source=ipl)
    comm.Barrier()

@autojit
def sim5pto(dx, vector, y0, ghost):

    integral = np.zeros_like(vector)

    for i in range(ghost,Nx+1):
    
        integral[i] = - 19*vector[i-2] + 346*vector[i-1] + 456*vector[i] - 74*vector[i+1] + 11*vector[i+2]
        
    if rank==size-1:
        integral[Nx-1] =  11*vector[Nx-4] -  74*vector[Nx-3] + 456*vector[Nx-2] + 346*vector[Nx-1] -  19*vector[Nx]
        integral[Nx]   = -19*vector[Nx-4] + 106*vector[Nx-3] - 264*vector[Nx-2] + 646*vector[Nx-1] + 251*vector[Nx]

    if rank==0:
        integral[ghost+1] = -19*vector[ghost+4] + 106*vector[ghost+3] - 264*vector[ghost+2] + 646*vector[ghost+1] + 251*vector[ghost]

    integral   *= dx/720

    if rank==0:
        integral[ghost]   = y0

    integral  = integral.cumsum()

    suma = np.zeros(1)

    # Los procesos que no son el 0 están bloqueados esperado la suma:
    if rank != 0:
        comm.Recv([suma, MPI.DOUBLE], source=ipl)

    # Inicialmente, solo el 0 llega hasta aquí desencadenado la cascada:
    integral += suma[0]

    comm.Send([integral[Nx:Nx+1],MPI.DOUBLE], dest=ipr)
    comm.Barrier()

    return integral

@autojit
def derivada1a(dx, vector, ghost):
    
    der = np.zeros_like(vector)

    # Simétrica en 5 puntos:
    for i in range(ghost,Nx+1):
        der[i] = vector[i-2] - 8*vector[i-1] + 8*vector[i+1] - vector[i+2] 
    
    if rank==0:
        
        der[ghost]   = -25*vector[ghost]   + 48*vector[ghost+1] - 36*vector[ghost+2] + 16*vector[ghost+3] - 3*vector[ghost+4]
        der[ghost+1] = - 3*vector[ghost]   - 10*vector[ghost+1] + 18*vector[ghost+2] -  6*vector[ghost+3] +   vector[ghost+4]

        
    if rank==size-1:
        
        der[Nx]   = 25*vector[Nx] - 48*vector[Nx-1] + 36*vector[Nx-2] - 16*vector[Nx-3] + 3*vector[Nx-4]        
        der[Nx-1] =  3*vector[Nx] + 10*vector[Nx-1] - 18*vector[Nx-2] +  6*vector[Nx-3] -   vector[Nx-4]

    der /= 12*dx
    
    return der

@autojit    
def derivada2a_boundary(dx, vector):
    
    ''' Función para calcular la derivada segunda en el boundary. '''

    der  =  45*vector[Nx] - 154*vector[Nx-1] + 214*vector[Nx-2] - 156*vector[Nx-3] + 61*vector[Nx-4] - 10*vector[Nx-5]
    der /= 12*dx**2

    return der    

@autojit
def disipacion8o_old(vector):
    
    disipacion = np.zeros_like(vector)
    
    # Generamos el vector disipación:
    for i in range(ghost,Nx+1):

        disipacion[i] = - vector[i+4] - vector[i-4] + 8*(vector[i+3] + vector[i-3])       \
                        - 28*(vector[i+2] + vector[i-2]) + 56*(vector[i+1] + vector[i-1]) \
                        - 70*vector[i]
    
    # Disipación en la frontera pi/2:
    ## De: Mattsson, K., Svärd, M. and Nordström, J., “Stable and Accurate Artificial Dissipation”, 
    ## J. Sci. Comput., 21
    if rank==size-1: 
        disipacion[Nx-3] =   4*vector[Nx] - 22*vector[Nx-1] + 52*vector[Nx-2] - 69*vector[Nx-3] + 56*vector[Nx-4] - 28*vector[Nx-5] + 8*vector[Nx-6] - vector[Nx-7]
        disipacion[Nx-2] = - 6*vector[Nx] + 28*vector[Nx-1] - 53*vector[Nx-2] + 52*vector[Nx-3] - 28*vector[Nx-4] +  8*vector[Nx-5] -   vector[Nx-6]
        disipacion[Nx-1] =   4*vector[Nx] - 17*vector[Nx-1] + 28*vector[Nx-2] - 22*vector[Nx-3] +  8*vector[Nx-4] -    vector[Nx-5]
        disipacion[Nx]   =   - vector[Nx] +  4*vector[Nx-1] -  6*vector[Nx-2] +  4*vector[Nx-3] -    vector[Nx-4]
    
    if rank==0:
        disipacion[ghost]    =   - vector[ghost] +  4*vector[ghost+1] -  6*vector[ghost+2] +  4*vector[ghost+3] -    vector[ghost+4]
        disipacion[ghost+1]    =   4*vector[ghost] - 17*vector[ghost+1] + 28*vector[ghost+2] - 22*vector[ghost+3] +  8*vector[ghost+4] -    vector[ghost+5]
        disipacion[ghost+2]    = - 6*vector[ghost] + 28*vector[ghost+1] - 53*vector[ghost+2] + 52*vector[ghost+3] - 28*vector[ghost+4] +  8*vector[ghost+5] -   vector[ghost+6]
        disipacion[ghost+3]    =   4*vector[ghost] - 22*vector[ghost+1] + 52*vector[ghost+2] - 69*vector[ghost+3] + 56*vector[ghost+4] - 28*vector[ghost+5] + 8*vector[ghost+6] - vector[ghost+7]

    disipacion *= Coef/dx
    
    return disipacion

@autojit
def disipacion8o(vector):
    
    disipacion = np.zeros_like(vector)
    
    # Generamos el vector disipación:
    for i in range(ghost,Nx+1):

        disipacion[i] = - vector[i+4] - vector[i-4] + 8*(vector[i+3] + vector[i-3])       \
                        - 28*(vector[i+2] + vector[i-2]) + 56*(vector[i+1] + vector[i-1]) \
                        - 70*vector[i]

    disipacion *= Coef/dx

    if rank==size-1:
        for i in range(ghost):
            disipacion[Nx-i] = 0.

    if rank==0:
        for i in range(ghost):
            disipacion[ghost+i] = 0.
    
    return disipacion

@autojit
def disipacion6o(vector):
    
    disipacion = np.empty_like(vector)
    
    # Generamos el vector disipación:
    for i in range(ghost,Nx+1):
        disipacion[i] = vector[i+3] - 6*(vector[i+2] + vector[i-2]) + 15*(vector[i+1] + vector[i-1]) - 20*vector[i] + vector[i-3]          

    disipacion *= Coef/dx

    return disipacion

@autojit
def delta_A_M_solver(Phi, Pi, x, t):

    rho = Phi**2 + Pi**2

    integrando = -np.sin(x)*np.cos(x)*rho
    delta      =  sim5pto(dx, integrando, 0, ghost)

    # Necesitamos el valor del delta y de Pi en el boundary:
    if rank == size-1:
        delta_boundary = delta[Nx:Nx+1]
        Pi_boundary    = Pi[Nx:Nx+1]

    else:
        delta_boundary = np.zeros(1)
        Pi_boundary    = np.zeros(1)

    comm.Bcast(delta_boundary, root=size-1)
    comm.Bcast(Pi_boundary,    root=size-1)

    delta         -= delta_boundary[0] # Elección del gauge
    delta_boundary = np.zeros(1)

    sync(delta)

    # Para calcular A, utilizamos las expresión (8) de las notas de Javier:
    # Primera parte de la integración:
    integrando  = rho*np.exp(-delta) - np.exp(-delta_boundary[0])*Pi_boundary[0]**2
    integrando *= tan2x

    # Valor regularizado del integrando calculado a partir de la expansión en serie:
    if rank == size-1:
        integrando[Nx] = 5*Pib**4/2
    
    integral    = sim5pto(dx, integrando, 0, ghost)

    # Calculamos la masa:
    if rank==size-1:
        M = np.exp(delta[Nx])*integral[Nx] - 0.5*np.pi*Pi[Nx]**2
    else:
        M = 0.

    # Primera parte de la integral:
    f           = np.exp(delta)*(np.cos(x)**3)*integral
    Omega       = np.sin(x)
    fovOmega    = f/Omega

    if rank==0:
        fovOmega[ghost] = 0. # Regularización en x=0

    A = 1 - fovOmega

    # Segunda parte de la integral (parte resuelta analíticamente):
    factor    = -x*np.cos(x)**3/np.sin(x) + np.cos(x)**2

    if rank==0:
        factor[ghost] = 0. # Regularización en x=0

    A -= np.exp(delta)*factor*np.exp(-delta_boundary[0])*Pi_boundary[0]**2

    sync(A)
    
    return delta, A, M

@autojit
def k_Phi_solver(A, delta, Phi, Pi, x, t):
    
    der        = np.empty_like(x)

    disipacion = disipacion8o(Phi)

    # El vector a derivar:
    vector = A*np.exp(-delta)*Pi
    der    = derivada1a(dx, vector, ghost)

    # Añadimos la disipación:
    der += disipacion
    
    return der

@autojit
def k_Pi_solver(A, delta, Phi, Pi, x, t):
    
    der        = np.empty_like(x)

    disipacion = disipacion8o(Pi)    
    
    f = A*Phi*np.exp(-delta)

    # La expresión siguiente tiene un problema para x=pi/2 y x=0 pero por razones distintas:
    vector = f*tan2x
    der    = derivada1a(dx, vector, ghost)
    der   *= tan2xI

    if rank==0:

        # Punto i=0:
        dPhi0    = 2*(8*Phi[ghost+1] - Phi[ghost+2]) # Por paridad
        dPhi0    = Phi[ghost-2] - 8*Phi[ghost-1] + 8*Phi[ghost+1] - Phi[ghost+2] 
        dPhi0   /= 12*dx

        # Corregimos la derivada con el valor regular
        der[ghost]   = d*dPhi0*np.exp(-delta[ghost])

    if rank==size-1:

        # Punto i=Nx:
        dPhiNx   = 0 # Forzamos el valor correcto en la frontera obtenido de la expansión en serie

        # Corregimos la derivada con el valor regular
        der[Nx]  = dPhiNx

    # Calculamos la derivada de f, que es perfectamente regular:
    df = derivada1a(dx, f, ghost)
    
    # El cociente tiene que ser regularizado en x=0, pi/2:
    fov_Omega     = f/Omega

    if rank==0:
        fov_Omega[ghost]  = (d-1)*dPhi0*np.exp(-delta[ghost])  # Regularización en x=0

    if rank==size-1:
        fov_Omega[Nx] = (d-1)*dPhiNx                   # Regularización en x=pi/2
    
    # Calculamos la derivada de f/Omega:
    dfov_Omega = derivada1a(dx, fov_Omega, ghost)
    
    if rank==size-1:

        #for j in range(2*ghost,Nx):
        for j in range(Nx-100,Nx):

            # Sustituimos la derivada en los puntos interiores:
            der[j] = df[j] + df[j]/dOmega[j] - dfov_Omega[j]*Omega[j]/dOmega[j]

    # Añadimos la disipación en las demás particiones:
    der += disipacion

    # En el caso con source necesitamos que la derivada en Nx sea nula:
#    if rank==size-1:
#        der[Nx] = 0.

    return der
    
@autojit
def MomentumConstraint(A, Ap, App, dt, Pi, Phi, delta):

    Adot  = App - 4*Ap + 3*A
    Adot /= 2*dt

    constraint = Adot + 2*np.exp(-delta)*np.sin(x)*np.cos(x)*Phi*Pi*A**2

    return constraint    

@autojit
def RK4Step(A, delta, Phi, Pi, x, Time, dt):

    # Paso 1:
    k_Phi1 = k_Phi_solver(A, delta, Phi, Pi, x, Time)
    k_Pi1  = k_Pi_solver (A, delta, Phi, Pi, x, Time)

    # Paso 2:
    Phi_int = Phi + 0.5*dt*k_Phi1
    Pi_int  = Pi  + 0.5*dt*k_Pi1

    BC(Phi_int, Pi_int, Time+0.5*dt)

    sync(Phi_int)
    sync(Pi_int)

    delta_int, A_int, aux = delta_A_M_solver(Phi_int, Pi_int, x, Time+0.5*dt)

    k_Phi2 = k_Phi_solver(A_int, delta_int, Phi_int, Pi_int, x, Time+0.5*dt)
    k_Pi2  = k_Pi_solver(A_int, delta_int, Phi_int, Pi_int, x, Time+0.5*dt)

    # Paso 3:
    Phi_int = Phi + 0.5*dt*k_Phi2
    Pi_int  = Pi  + 0.5*dt*k_Pi2

    BC(Phi_int, Pi_int, Time+0.5*dt)

    sync(Phi_int)
    sync(Pi_int)

    delta_int, A_int, aux = delta_A_M_solver(Phi_int, Pi_int, x, Time+0.5*dt)

    k_Phi3 = k_Phi_solver(A_int, delta_int, Phi_int, Pi_int, x, Time+0.5*dt)
    k_Pi3  = k_Pi_solver(A_int, delta_int, Phi_int, Pi_int, x, Time+0.5*dt)

    # Paso 4:
    Phi_int = Phi + dt*k_Phi3
    Pi_int  = Pi  + dt*k_Pi3

    BC(Phi_int, Pi_int, Time+dt)

    sync(Phi_int)
    sync(Pi_int)

    delta_int, A_int, aux = delta_A_M_solver(Phi_int, Pi_int, x, Time+dt)

    k_Phi4 = k_Phi_solver(A_int, delta_int, Phi_int, Pi_int, x, Time+dt)
    k_Pi4  = k_Pi_solver(A_int, delta_int, Phi_int, Pi_int, x, Time+dt)

    # Actualizamos los vectores:
    Phi += (k_Phi1 + 2*(k_Phi2 + k_Phi3) + k_Phi4)*dt/6
    Pi  += (k_Pi1  + 2*(k_Pi2  + k_Pi3 ) + k_Pi4 )*dt/6

    BC(Phi, Pi, Time+dt)

    sync(Phi)
    sync(Pi)

    return Phi, Pi


if __name__ == '__main__':

    # Parseamos la linea de comandos:
    parser = argparse.ArgumentParser()

    parser.add_argument('-a','--alpha',help='Parámetro alpha',type=float)

    args   = parser.parse_args()
    alpha  = args.alpha

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Calculamos los ids de los procesos izq. y der. del actual:
    ipr = rank + 1
    ipl = rank - 1

    if ipr==size:
        ipr = MPI.PROC_NULL

    if ipl<0:
        ipl = MPI.PROC_NULL

    # Coeficiente de disipación:
    Coef   = 0.00001
    Coef   = 0.005
    Coef   = 0.0

    d      = 3       # Dimensión de AdS_d+1

    # Número de puntos de la simulacion:
    points     = 2**10

    # Archivos de salida:
    fout_evolucion = 'DD_evolucion_Source_estatico_%4.2fx%i.%i.nc' % (alpha, points, rank)
    fout_series    = 'DD_series_Source_estatico_%4.2fx%i.%i.nc' % (alpha, points, rank)

    # Número de segmentos de la malla:
    segments   = points - 1

    # Número de ghost points
    ghost  = 4

    # Tamaño de las particiones:
    chunk_size = points/size

    # Índices inicial y final de las particiones:
    slices     = [(i*chunk_size,(i+1)*chunk_size+2*ghost) for i in range(size)]

    # Vector x:
    dx = 0.5*np.pi/segments
    x  = dx*np.arange(rank*chunk_size - ghost, (rank+1)*chunk_size + ghost)

    # Calculamos el paso temporal:
    Courant = 0.1
    dt      = Courant*dx

    if rank==0:
        print(" Corriendo en %d cores" % size)
        print("alpha = %4.2f" % alpha)
        print("dt    = %4.2e" % dt)

    Nx, = x.shape 
    Nx -= (ghost + 1) # Corregimos la dimensión en el número de ghost points

    tan2x    = np.tan(x)**2
    sinx     = np.sin(x)
    cosx     = np.cos(x)

    if rank == size-1:
        tan2x[Nx] = np.inf
        sinx[Nx] = 1.
        cosx[Nx] = 0.

    tan2xI     = 1/tan2x

    Omega  = sinx*cosx/(d-1) 
    dOmega = (cosx**2 - sinx**2)/(d-1)

    # Configuración a t=0:
    Time = 0.

    # Esta es una forma sencilla de construir los initial data:
    i,f = slices[rank]
    Pi  = InitialPi_new(alpha, points)[i:f]
    Phi = np.zeros_like(Pi)

    # Solo el proceso rank+1 necesita el valor de Pi en el boundary:
    if rank==size-1:
        Pib = Pi[Nx]

    # Cogemos la solución calculada con las derivadas centradas a 2**12 puntos:
#    Pi  = InitialPi_nc('tmp/DC_evolucion_Source_estatico_2.71x4096.nc')[i:f]
#    Phi = InitialPhi_nc('tmp/DC_evolucion_Source_estatico_2.71x4096.nc')[i:f]

    # inicializamos los buffers de ghost points:
    BC(Phi, Pi, Time)

    # Datos iniciales para A y delta:
    delta, A, M  = delta_A_M_solver(Phi, Pi, x, Time)
    buffer_tmp   = np.zeros_like(A)
    
    comm.Allreduce(A, buffer_tmp, op=MPI.MIN)
    Amin = buffer_tmp[ghost:Nx+1].min()

    # Guardamos las configuraciones a t=0:
    buffer_series = {'time' : [], 'Amin' : [], 'Masa' : [], 'Pi0' : [], 'VeV' : []}

    buffer_series['time'].append(Time)
    buffer_series['Amin'].append(Amin)
    buffer_series['Masa'].append(M)
    buffer_series['Pi0' ].append(Pi[Nx])
    buffer_series['VeV' ].append(derivada2a_boundary(dx, Phi))
    
    # Variables para calcular el MomentumConstraint:        
    Ap         = np.zeros(A.shape)
    App        = np.zeros(A.shape)
    constraint = np.zeros(A.shape)    

    crea_evolucion(fout_evolucion, x)
    buffer_out(fout_evolucion, Time, Phi, Pi, A, delta, constraint)
    crea_series(fout_series)

    # Control del bucle principal:
    EndTime = 200.
    Nt      = int(EndTime/dt) # Número de iteraciones
    tSave   = 0.1 
    Nsave   = int(tSave/dt)

    for i in range(1,Nt):
        
        # Guardamos el paso previo antes de sobrescribir:
        App = Ap.copy()
        Ap  = A.copy()        

        # Avanzamos la solución en el tiempo:
        Phi, Pi     = RK4Step(A, delta, Phi, Pi, x, Time, dt)
        delta, A, M = delta_A_M_solver(Phi, Pi, x, Time+dt)

        Time += dt
    
        # ALLreduce para el mínimo de A    
        comm.Allreduce(A, buffer_tmp, op=MPI.MIN)
        Amin = buffer_tmp[ghost:Nx+1].min()

        # Guardamos las series en el buffer:
        buffer_series['time'].append(Time)
        buffer_series['Amin'].append(Amin)
        buffer_series['Masa'].append(M)
        buffer_series['Pi0' ].append(Pi[Nx])
        buffer_series['VeV' ].append(derivada2a_boundary(dx, Phi))
        
        if (np.mod(i,Nsave)==0) or (Amin<0):
            
            # Calculo del MomentumConstraint:
            constraint = MomentumConstraint(A, Ap, App, dt, Pi, Phi, delta)            

            buffer_out(fout_evolucion, Time, Phi, Pi, A, delta, constraint)
            escribe_series(fout_series, buffer_series)    
            
            # Reset del buffer:
            buffer_series = {'time' : [], 'Amin' : [], 'Masa' : [], 'Pi0' : [], 'VeV' : []}

            if rank==0:

                print '--> Iteracion %i de %i Nt. t=%f' % (i,Nt,Time)
                
        # Si se produce el colapso, A se va por debajo de cero:
        if Amin<0:
            
            break
