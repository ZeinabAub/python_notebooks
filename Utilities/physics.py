# Class for a moving body, with classes for getting the energy and the plots in a simple way
import numpy as np
from scipy.integrate import solve_ivp

class mover: #Class for a generic moving body
  
  def __init__(self, mass = 1.0, q0 =[ 0.0], v0 = [0.0], name = '', size=0.0):
    self.mass = mass
    self.Q = np.array([q0])  #list of positions
    self.V = np.array([v0])  #list of velocities
    self.P = self.V*self.mass
    self.N = 1  #number of positions in list
    self.d = len(q0)  #number of dimensions of the system
    self.t0 = 0.0
    self.tf = 1.0
    self.T = np.array([self.t0, self.tf]) #array of times of the system
    self.energy = np.zeros(2)
    self.kinetic_energy = np.zeros(2)
    self.marker = None
    self.trace_plot = None
    self.name = name
    self.size = size
    
  def __str__(self):  #Print the body info
    return f'Body with mass {self.mass}, position {self.Q[-1]}, velocity {self.V[-1]}'
    
  
  def q(self, i = None): #return positions at the i-esime instant
    return self.Q[-1] if i == None else self.Q[i]  #As default the last position is returned
  
  def set_q(self, q): #set positions
    self.Q = q
    self.N = len(q)
  
  def v(self, i = None): #return velocities
    return self.V[-1] if i == None else self.V[i] 
  
  def set_v(self, v): #set velocities
    self.V = v
    self.N = len(v)
    
  def p(self, i = None): #return momentums
    return self.P[-1] if i == None else self.P[i] 
  
  def set_p(self, p): #set momentums
    self.P = p
    self.N = len(p)
  
  def set_T(self, T):
    self.T = T
    self.N = len(T)
  
  def K(self, i):
    return 0.5*self.mass*self.v(i)**2
  
  def set_energy(self, f):
    self.energy = np.zeros(self.N)
    for i in range(self.N):
      self.energy[i] = f(self.Q[i], self.V[i], self.mass)
    
  def E(self, i):
    return self.energy[i]

  def trace(self, start = None, len = None):
    if start == None: start=self.N
    if len == None: len = self.N 
    end = 0 if start<len else start-len
    return np.transpose(self.Q[end:start])
  
  def v_tail(self, start = None, len = None):
    if start == None: start=self.N
    if len == None: len = self.N 
    end = 0 if start<len else start-len
    return np.transpose(self.V[end:start])
  
  def solve_ODE(self, f, T): #Solve the ODE after setting f(q,v)      
      y0 = np.append(self.q(), self.v())
      Y = solve_ivp(f, (T[0], T[-1]), y0, 'RK45', t_eval=T)
      self.set_T(T)
      self.set_q(np.transpose(Y.y[0:self.d]))
      self.set_v(np.transpose(Y.y[self.d: 2*self.d]))
     
def handle_collisions(bodies):
  N = len(bodies)
  for i in range(N-1):
    for j in range(i+1,N):
      a, b = bodies[i], bodies[j]
      if(np.linalg.norm(a.q-b.q)<=(a.size+b.size)):
        r = a.q-b.q
        u_r = r/np.linalg.norm(r)
        v_a, v_b = a.v, b.v
        a.v = v_a - 2*b.mass/(a.mass+b.mass)*np.dot((v_a-v_b),u_r)*u_r
        b.v = v_b - 2*a.mass/(a.mass+b.mass)*np.dot((v_b-v_a),u_r)*u_r