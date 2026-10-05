from __future__ import annotations

def rk4_step(func,x,t,dt):
    k1=func(t,x); k2=func(t+dt/2,x+dt*k1/2); k3=func(t+dt/2,x+dt*k2/2); k4=func(t+dt,x+dt*k3)
    return x+dt*(k1+2*k2+2*k3+k4)/6

def integrate_ode(func,x0,t0=0.0,t1=1.0,steps=32):
    if steps<1: raise ValueError('steps must be positive')
    dt=(t1-t0)/steps; x=x0; t=float(t0)
    for _ in range(steps): x=rk4_step(func,x,t,dt); t+=dt
    return x
