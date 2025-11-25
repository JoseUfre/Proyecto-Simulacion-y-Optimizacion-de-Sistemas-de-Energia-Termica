import CoolProp.CoolProp as cp
import numpy as np


#Parámetros fijos:

c_kwh = 0.12 #CLP/kWh, valor estándar aprox para industrias
#Funciones de costos fijos de equipos
h = 5000 #Horas anuales de funcionamiento
n = 5 #años a considerar

def C_comp_low(W):
    return 10167.5 * (W* 1e-3)**0.46 

def C_comp_high(W):
    return 9624.2 * (W* 1e-3)**0.46

def C_valve(m_dot):
    return 114.5*m_dot

def C_ev_cond(A):
    return 1397*A**0.89

def C_HX_cascade(A):
    return 383.5*A**0.65

#Funciones

#do debe poder definirse en torno a los valores de di y t.
def do(di, t):
    return di + 2*t

#El paso pt se define igual como una función.
def pt(do):
    return 1.25 * do

def epsilon(m_h, m_c, cp_h, cp_c, U, A):
    C_h = m_h * cp_h
    C_c = m_c * cp_c
    C_min = min(C_h, C_c)
    C_max = max(C_h, C_c)
    C_r = C_min / C_max
    NTU = U * A / C_min

    #Eficiencia del intercambiador para intercambiador de doble paso y flujo cruzado:
    epsilon = (1 - np.exp(-NTU * (1 - C_r))) / (1 - C_r * np.exp(-NTU * (1 - C_r)))

#Función para la exergía específica
def b(T, cp, T_0): 
    return cp * (T - T_0 - T_0 * np.log(T / T_0))

#Velocidad en los tubos
def v_tubo(m, di, Np, Nt, p_c):
    A_c = (np.pi * di**2) / 4
    v = m / (p_c * A_c * Nt / Np)
    return v

#Reynolds para los tubos
def Reynolds(di, mu, v, L, p):
    A_c = (np.pi * di**2) / 4 #Área de un tubo
    Re = (p * v * di) / mu
    return Re

#h_i para los tubos
def h_i(Re, Pr, di, k_c): #Dittus-Boelter: Retorna h_i
    return (0.023 * Re**0.8 * Pr**0.4)*k_c/di

#h_o para la carcasa
def h_o(Re, Pr, D_e, k_h): #Kern: Retorna h_o
    return (0.36 * Re**0.55 * Pr**0.33)*k_h/D_e

#U_global
def U_global(h_i, h_o, d_i, d_o, Rf_i, Rf_o, k_w):
    R_i = d_o / (h_i * d_i)
    R_o = 1 / (h_o)
    R_w = d_o * np.log(d_o/d_i) / (2 * k_w)
    R_fi = Rf_i * d_o / (d_i)
    R_fo = Rf_o
    
    U = 1 / (R_i + R_o + R_w + R_fi + R_fo)
    return U

#Pérdidas de fricción en los tubos
def f_i_tubo(Re):
    return 0.316 * Re**(-0.25)

#Pérdidas de fricción en la carcasa
def f_i_carcasa(Re):
    return 0.14 * Re**(-0.15)

#Pérdidas de presión en los tubos
def P_tubos(Np, d_i, f_i, rho_c, v_i, L):
    return Np * f_i * L  *rho_c * v_i ** 2 / (2*d_i)

#Pérdidas de presión en la carcasa
def P_carcasa(f_s, G, rho_h, L, D_e):
    return f_s * (G ** 2) * L / (2 * rho_h * D_e)

#Diámetro equivalente
def D_e(pt, d_o):
    return 1.27 * (pt ** 2 - 0.785 * d_o ** 2) / d_o

#Velocidad másica en la carcasa
def Vm_carcasa(A_cr, m_h):
    return m_h / A_cr

#Reynolds en la carcasa
def Re_s(G, De, mu):
    return G * De / mu

#Diámetro de carcasa
def D_s(Nt, pt):
    return pt * (Nt * 3 ** 0.5 / np.pi) ** 0.5

#Potencia de la bomba
def W_pump(m, p, Delta_P, eta_s):
    return m * Delta_P / (eta_s * p)

#Área de intercambio
def A_hx(L, Nt, do):
    return np.pi * do * L * Nt


def delta_TM(T_h_in, T_c_in, T_h_out, T_c_out):
    deltaT1 = T_h_in - T_c_out
    deltaT2 = T_h_out - T_c_in
    try:
        return (deltaT1 - deltaT2)/(np.log((deltaT1)/(deltaT2)))
    except (ZeroDivisionError, ValueError, RuntimeWarning):
        return (deltaT1 + deltaT2)*0.5


#Clases
#Clase para definir el refrigerante
class refrigerante():
    def __init__(self, GWP=None, T_cond=None, T_eb=None, rho=None, cp=None, mu=None, k=None):
        self.GWP = GWP
        self.T_cond = T_cond
        self.T_eb = T_eb
        self.rho = rho #densidad
        self.cp = cp #Calor específico
        self.mu = mu
        self.k = k

#Clase para definir el HX
class HX():
    def __init__(self, U, pinch, fluid_c, fluid_h, Np, Nt, di, t, m_h=None, m_c=None, T_c_in=None, T_h_in=None, tipo=None):

        self.tipo = tipo # 'evaporador', 'condensador' o 'mixto'

        self.T_c_in = T_c_in
        self.T_h_in = T_h_in
        self.pinch = pinch

        self.fluid_c = fluid_c
        self.fluid_h = fluid_h

        self.m_c = m_c
        self.m_h = m_h
        self.Np = Np
        self.Nt = Nt
        self.di = di
        self.t = t
        self.do = do(self.di, self.t)
        self.L = None

        if self.tipo == 'evaporador':
            self.mu_h = self.fluid_h.mu
            self.p_h = self.fluid_h.rho
            self.cp_h = self.fluid_h.cp
            self.k_h = self.fluid_h.k

        elif self.tipo == 'condensador':
            self.mu_c = self.fluid_c.mu
            self.p_c = self.fluid_c.rho
            self.cp_c = self.fluid_c.cp
            self.k_c = self.fluid_c.k

        self.U = U

        self.costo = None #Costo capital del equipo


        self.eta_II_v = None
        self.Ex_dest_v = None
        self.C_bomb = None
        self.A = None
        self.L = None
        self.epsilon = None


    def solve(self, T_chamber, T_cond, T_0, m_h, m_c, s2=None, s3=None, h2=None, h3=None, Tsh=None, Tsc=None, Q=None, Q_c=None): 
        #h2, h3, s2 y s3 para resolver el condensador

        d_o = do(self.di, self.t)
        #p_t = pt(d_o)


        #A_v = A_hx(self.L, self.Nt, d_o) No se define directamente, depende de la carga Q
        if self.tipo == 'evaporador':
            self.T_h_in = T_chamber + Q/(m_h * self.cp_h)

            T_ev_low = T_chamber - self.pinch
            DT1 = self.T_h_in - T_ev_low
            DT2 = T_chamber - T_ev_low
            #self.T_lm = (DT1 - DT2)/np.log(DT1/DT2)
            self.T_lm = delta_TM(self.T_h_in, self.T_c_in, T_chamber, T_ev_low)
            self.A = Q/(self.U*self.T_lm)
            self.L = self.A / (self.Nt * self.do * np.pi)

            C_h_v = self.m_h * self.cp_h
            h1 = cp.PropsSI("H", "T", self.T_c_in, "Q", 1, self.fluid_c)
            h4 = cp.PropsSI("H", "T", T_cond, "Q", 1, self.fluid_c)
            C_c_v = h1 - h4

            b_out_h = b(T_chamber, self.cp_h, T_0)

            P_e = cp.PropsSI("P", "T", T_ev_low, "Q", 1, self.fluid_c)
            b_out_c = cp.PropsSI("S", "T", T_ev_low + Tsh, "P", P_e, self.fluid_c)
            
            b_in_h = b(self.T_h_in, self.cp_h, T_0)
            h4 = cp.PropsSI("H", "T", T_cond, "Q", 1, self.fluid_c)
            b_in_c = cp.PropsSI("S", "P", P_e, "H", h4, self.fluid_c)

        elif self.tipo == 'condensador': #En este caso, el fluido frío será aire
            self.T_c_out = T_0 + Q_c/(m_c * self.cp_c)
            T_cond_high = self.T_c_out + self.pinch
            DT1 = self.T_c_in - T_cond_high
            DT2 = self.T_c_out - T_cond_high
            self.T_lm = abs((DT1 - DT2)/np.log(DT1/DT2))
            self.A = Q_c/(self.U*self.T_lm)
            self.L = self.A / (self.Nt * self.do * np.pi)

            C_c_v = m_c * (self.T_c_out - self.T_c_in)
            C_h_v = h2 - h3

            b_out_h = s3
            b_in_h = s2
            b_out_c = b(self.T_c_out, self.cp_c, T_0)
            b_in_c = 0 #Ya que es aire a T ambiente

        
        C_min_v = min(C_h_v, C_c_v)
        C_max_v = max(C_h_v, C_c_v)
        C_r_v = C_min_v / C_max_v
        
        #NTU_v = U_g * self.A / C_min_v
        NTU_v = self.U * self.A /C_min_v
        
        self.epsilon = (1 - np.exp(-NTU_v * (1 - C_r_v))) / (1 - C_r_v * np.exp(-NTU_v * (1 - C_r_v)))

        E_h_v = m_h * (b_in_h - b_out_h)
        E_c_v = m_c * (b_out_c - b_in_c)
        
        #Eficiencia y exergía destruida
        self.eta_II_v = E_c_v / E_h_v
        self.Ex_dest_v = E_h_v - E_c_v

        #Se define el costo capital del equipo
        self.costo = C_ev_cond(self.A)

class compresor():
    def __init__(self, eta_s, W, N):
        self.eta_s = eta_s
        self.W = W
        self.N = N

class VCR_simple():
    def __init__(self, Q, T_ev, T_cond, ref, eta_c, Tsh, Tsc, T_amb):
        self.Q = Q
        self.T_ev = T_ev
        self.T_cond = T_cond
        self.T_amb = T_amb
        self.fluid = ref
        self.eta_c = eta_c
        self.Tsh = Tsh
        self.Tsc = Tsc
        self.COP = None
        self.m_dot = None
        self.W_comp = None
        self.Qc = None

    def ciclo(self):
        #Estado 1: Entrada del compresor
        T1 = self.T_ev + self.Tsh
        Pe = cp.PropsSI("P", "T", self.T_ev, "Q", 1, self.fluid)
        s1 = cp.PropsSI("S", "T", T1, "P", Pe, self.fluid)
        h1 = cp.PropsSI("H", "T", T1, "P", Pe, self.fluid)

        #Estado 2: Salida del compresor
        s2s = s1 #Compresor "isentrópico"
        Pc = cp.PropsSI("P", "T", self.T_cond, "Q", 1, self.fluid) #Presión en el condensador
        T2s = cp.PropsSI("T", "S", s2s, "P", Pc, self.fluid)
        h2s = cp.PropsSI("H", "P", Pc, "T", T2s, self.fluid)
        h2 = h1 + (h2s - h1)/self.eta_c
        T2 = cp.PropsSI("T", "P", Pc, "H", h2, self.fluid)
        s2 = cp.PropsSI("H", "T", T2, "P", Pc, self.fluid)

        #Estado 3: Salida del condensador/entrada de la válvula
        T3 = self.T_cond - self.Tsc
        h3 = cp.PropsSI("H", "P", Pc, "T", T3, self.fluid)
        s3 = cp.PropsSI("S", "P", Pc, "T", T3, self.fluid)

        #Estado 4: Salida de válvula/entrada del condensador
        h4 = h3
        s4 = cp.PropsSI("S", "P", Pe, "H", h4, self.fluid)
        T4 = cp.PropsSI("T", "P", Pe, "H", h4, self.fluid)
        Q4 = cp.PropsSI("Q", "P", Pe, "H", h4, self.fluid) #Adicionalmente, se puede obtener la calidad

        #Ahora, obtenemos parámetros como el flujo másico y de potencia
        self.m_dot = self.Q/(h1 - h4)
        self.W_comp = self.m_dot*(h2 - h1)
        self.Qc = self.m_dot*(h2 - h3)
        self.COP = self.Q/self.W_comp 


#Clase para el ciclo en cascada
class VCR_cascada():
    def __init__(self, Q, T_ev_low, T_cond_low, T_ev_high, T_cond_high, ref1, ref2,
                 eta_c_low, eta_c_high, Tsh, Tsc, T_amb, T_chamber, m_g_e_low, m_aire, 
                 ref_e_low, U_e_low, HX_e_low, HX_c_high, HX_cascade=None):
        self.Q = Q
        self.Q_hx = None #Calor intercambiado en HX

        self.T_ev_low = T_ev_low #Temperatura del evaporador de baja temperatura
        self.T_cond_low = T_cond_low #Temperatura del condensador de baja temperatura
        self.T_ev_high = T_ev_high #Temperatura del evaporador de alta temperatura
        self.T_cond_high = T_cond_high #Temperatura del condensador de alta temperatura
        self.T_amb = T_amb #Temperatura ambiental

        self.fluid_low = ref1 #Fluido en ciclo de baja temperatura
        self.fluid_high = ref2 #Fluido en ciclo de alta temperatura

        self.eta_s_low = eta_c_low #Eficiencia 
        self.eta_s_high = eta_c_high

        self.Tsh = Tsh
        self.Tsc = Tsc

        #Para el evaporador de baja presión (HX)
        self.m_g = m_g_e_low #fluido másico de glicol en ev. baja T
        self.m_aire = m_aire #fluido másico de aire en cond. alta T
        self.T_chamber = T_chamber #Temperatura de espacio a refrigerar
        self.rho_e_low = ref_e_low.rho
        self.cp_e_low = ref_e_low.cp
        self.T_e_low_in = None
        self.T_lm_e_low = None
        self.T_lm_c_high = None

        self.U_e_low = U_e_low
        self.A_e_low = None
        self.A_c_high = None
        self.A_HX = None


        self.Pe_low = None
        self.Pc_low = None
        self.Pe_high = None
        self.Pc_high = None
        
        self.COP = None

        self.m_dot_low = None
        self.W_comp_low = None
        self.Qc_low = None
        self.m_dot_high = None
        self.W_comp_high = None
        self.Qc_high = None
        self.costo = None

        #Exergía destruida
        self.Ex_d_comp_l = None
        self.Ex_d_comp_h = None
        self.Ex_d_valve_l = None
        self.Ex_d_valve_h = None
        self.Ex_d_ev = None
        self.Ex_d_cond = None
        self.Ex_d_total = 0
        self.Ex_d_HX = 0

        self.HX_e_l = HX_e_low
        self.HX_c_h = HX_c_high
        self.HX_cascade = HX_cascade

        self.ciclo()
        #self.HX_e_low()
        self.HX_e_l.solve(Q=self.Q, T_chamber=self.T_chamber, T_cond=T_cond_low, Tsh=self.Tsh, m_c=self.m_dot_low, m_h=self.m_g, T_0=self.T_amb)
        self.HX_c_h.solve(Q_c=self.Qc_high, s2=self.s2, s3=self.s3, h2=self.h2, h3=self.h3, T_chamber=self.T_amb, T_cond=T_cond_high,
                          Tsc=self.Tsc, m_h=self.m_dot_high, m_c=self.m_aire, T_0=self.T_amb)
    
        #self.HX_cascade.solve(Q=self.Q, T_chamber=self.T_chamber, T_cond=T_cond_low, Tsh=self.Tsh, m_c=self.m_dot_low, T_0=self.T_amb)

        self.T_lm_e_low = self.HX_e_l.T_lm
        self.A_e_low = self.HX_e_l.A

        self.T_lm_c_high = self.HX_c_h.T_lm
        self.A_c_high = self.HX_c_h.A

        self.Ex_d_ev = abs(self.HX_e_l.Ex_dest_v)
        self.Ex_d_cond = abs(self.HX_c_h.Ex_dest_v)

        self.Ex_d_total += self.Ex_d_ev + self.Ex_d_cond + self.Ex_d_HX

        self.costos()



    def ciclo(self):
        #Ciclo de baja presión
        #Estado 1: Entrada del compresor de baja presión
        T1 = self.T_ev_low + self.Tsh
        self.Pe_low = cp.PropsSI("P", "T", self.T_ev_low, "Q", 1, self.fluid_low)
        s1 = cp.PropsSI("S", "T", T1, "P", self.Pe_low, self.fluid_low)
        h1 = cp.PropsSI("H", "T", T1, "P", self.Pe_low, self.fluid_low)

        #Estado 2: Salida del compresor
        s2s = s1 #Compresor "isentrópico"
        self.Pc_low = cp.PropsSI("P", "T", self.T_cond_low, "Q", 1, self.fluid_low) #Presión en el condensador
        T2s = cp.PropsSI("T", "S", s2s, "P", self.Pc_low, self.fluid_low)
        h2s = cp.PropsSI("H", "P", self.Pc_low, "T", T2s, self.fluid_low)
        self.h2 = h1 + (h2s - h1)/self.eta_s_low
        T2 = cp.PropsSI("T", "P", self.Pc_low, "H", self.h2, self.fluid_low)
        self.s2 = cp.PropsSI("S", "T", T2, "P", self.Pc_low, self.fluid_low)

        #Estado 3: Salida del condensador/entrada de la válvula
        T3 = self.T_cond_low - self.Tsc
        self.h3 = cp.PropsSI("H", "P", self.Pc_low, "T", T3, self.fluid_low)
        self.s3 = cp.PropsSI("S", "P", self.Pc_low, "T", T3, self.fluid_low)

        #Estado 4: Salida de válvula/entrada del evaporador
        h4 = self.h3
        s4 = cp.PropsSI("S", "P", self.Pe_low, "H", h4, self.fluid_low)
        T4 = cp.PropsSI("T", "P", self.Pe_low, "H", h4, self.fluid_low)
        Q4 = cp.PropsSI("Q", "P", self.Pe_low, "H", h4, self.fluid_low) #Adicionalmente, se puede obtener la calidad

        #Ahora, obtenemos parámetros como el flujo másico y de potencia del ciclo de baja presión
        self.m_dot_low = self.Q/(h1 - h4)
        self.W_comp_low = self.m_dot_low*(self.h2 - h1)
        self.Qc_low = self.m_dot_low*(self.h2 - self.h3)


        #Ciclo de alta presión
        #Estado 1: Entrada del compresor de baja presión
        T5 = self.T_ev_high + self.Tsh
        self.Pe_high = cp.PropsSI("P", "T", self.T_ev_high, "Q", 1, self.fluid_high)
        s5 = cp.PropsSI("S", "T", T5, "P", self.Pe_high, self.fluid_high)
        h5 = cp.PropsSI("H", "T", T5, "P", self.Pe_high, self.fluid_high)

        #Estado 2: Salida del compresor
        s6s = s5 #Compresor "isentrópico"
        self.Pc_high = cp.PropsSI("P", "T", self.T_cond_high, "Q", 1, self.fluid_high) #Presión en el condensador
        T6s = cp.PropsSI("T", "S", s6s, "P", self.Pc_high, self.fluid_high)
        h6s = cp.PropsSI("H", "P", self.Pc_high, "T", T6s, self.fluid_high)
        h6 = h5 + (h6s - h5)/self.eta_s_high
        T6 = cp.PropsSI("T", "P", self.Pc_high, "H", h6, self.fluid_high)
        s6 = cp.PropsSI("S", "T", T6, "P", self.Pc_high, self.fluid_high)

        #Estado 3: Salida del condensador/entrada de la válvula
        T7 = self.T_cond_high - self.Tsc
        h7 = cp.PropsSI("H", "P", self.Pc_high, "T", T7, self.fluid_high)
        s7 = cp.PropsSI("S", "P", self.Pc_high, "T", T7, self.fluid_high)

        #Estado 4: Salida de válvula/entrada del evaporador
        h8 = h7
        s8 = cp.PropsSI("S", "P", self.Pe_high, "H", h8, self.fluid_high)
        T8 = cp.PropsSI("T", "P", self.Pe_high, "H", h8, self.fluid_high)
        Q8 = cp.PropsSI("Q", "P", self.Pe_high, "H", h8, self.fluid_high) #Adicionalmente, se puede obtener la calidad

        #Ahora, obtenemos parámetros como el flujo másico y de potencia del ciclo de baja presión
        self.Q_hx = self.Qc_low #Calor en intercambiador es el mismo en Ev. y Cond.
        self.m_dot_high = self.Q_hx/(h5 - h8)
        self.W_comp_high = self.m_dot_high*(h6 - h5)
        self.Qc_high = self.m_dot_high*(h6 - h7)

        self.COP = self.Q/(self.W_comp_low + self.W_comp_high)

        #Análisis exergético
        self.Ex_d_comp_l = self.T_amb * self.m_dot_low * (self.s2 - s1)
        self.Ex_d_comp_h = self.T_amb * self.m_dot_high * (s6 - s5)
        self.Ex_d_valve_l = self.T_amb * self.m_dot_low * (s4 - self.s3)
        self.Ex_d_valve_h = self.T_amb * self.m_dot_high * (s8 - s7)
        self.Ex_d_total = self.Ex_d_comp_l + self.Ex_d_comp_h + self.Ex_d_valve_l + self.Ex_d_valve_h

        #Intercambiador de en medio
        Ex_d_cond_m = (self.s2 - self.s3) * self.m_dot_low * self.T_amb
        Ex_d_ev_m = (s8 - s5) * self.m_dot_high * self.T_amb

        self.Ex_d_HX = Ex_d_cond_m + Ex_d_ev_m
        DT1 = self.T_cond_low - self.T_ev_high
        DT2 = self.T_cond_low - (self.T_ev_high + self.Tsh)  # Aproximación
        #self.T_lm_HX = (DT1 - DT2) / np.log(DT1/DT2) if DT1 != DT2 else DT1
        self.T_lm_HX = delta_TM(self.T_cond_low, self.T_ev_high, self.T_cond_low, self.T_ev_high + self.Tsh)
        self.A_HX = self.Q_hx / (self.U_e_low * self.T_lm_HX)


        


    def costos(self):
        #De compresores
        self.C_comp_l = C_comp_low(self.W_comp_low)/n + self.W_comp_low/1000*h * c_kwh

       
        self.C_comp_h = C_comp_high(self.W_comp_high)/n + self.W_comp_high/1000*h * c_kwh

        #De intercambiadores
        self.C_ev = self.HX_e_l.costo
        self.C_cond = self.HX_c_h.costo

        #De válvulas
        self.C_valve_l = C_valve(self.m_dot_low)
        self.C_valve_h = C_valve(self.m_dot_high)

        self.costo = self.C_comp_h + self.C_comp_l + self.C_ev + self.C_cond + self.C_valve_l + self.C_valve_h


    def HX_e_low(self):
        self.T_e_low_in = self.T_chamber + self.Q/(self.m_g * self.cp_e_low)

        self.U_e_low = 500 #Caso fijo
        DT1 = self.T_e_low_in - self.T_ev_low
        DT2 = self.T_chamber - self.T_ev_low
        self.T_lm_e_low = (DT1 - DT2)/np.log(DT1/DT2)
        self.A_e_low = self.Q/(self.U_e_low*self.T_lm_e_low)


    def show(self):
        print("\n" + "="*50)
        print("          Resultados del Ciclo en Cascada")
        print("="*50)

        print(f"{'COP total:':25s} {self.COP:10.3f}")
        print(f"{'Costo total:':25s} {self.costo/1000/n:.2f} mil USD (por año)")

        print("\n--- Ciclo de Baja Temperatura ---")
        print(f"{'m_dot_low [kg/s]:':25s} {self.m_dot_low:.4f}")
        print(f"{'W_comp_low [kW]:':25s} {self.W_comp_low/1000:.3f}")
        print(f"{'Qc_low [kW]:':25s} {self.Qc_low/1000:.3f}")
        print(f"{'Pe_low [kPa]:':25s} {self.Pe_low/1000:.3f}")
        print(f"{'Pc_low [kPa]:':25s} {self.Pc_low/1000:.3f}")

        print("\n--- Ciclo de Alta Temperatura ---")
        print(f"{'m_dot_high [kg/s]:':25s} {self.m_dot_high:.4f}")
        print(f"{'W_comp_high [kW]:':25s} {self.W_comp_high/1000:.3f}")


        print("="*50 + "\n")

        #Resultados en evaporador de baja temperatura
        print("Resultados en Evaporador de baja Temperatura")
        print("="*50)


        print("\n--- Parámetros del evaporador ---")
        print(f"{'Área [m]:':25s} {self.A_e_low:.3f}")
        print(f"{'T_lm [K]:':25s} {self.T_lm_e_low:.3f}")


        print("="*50 + "\n")

        #Resultados en condensador de alta temperatura
        print("Resultados en Condensador de alta Temperatura")
        print("="*50)


        print("\n--- Parámetros del condensador ---")
        print(f"{'Área [m]:':25s} {self.A_c_high:.3f}")
        print(f"{'T_lm [K]:':25s} {self.T_lm_c_high:.3f}")

        print("="*50 + "\n")

        #Resultados en HX
        print("Resultados en HX en cascada")
        print("="*50)


        print("\n--- Parámetros del HX en cascada ---")
        print(f"{'Área [m]:':25s} {self.A_HX:.3f}")


        print("="*50 + "\n")

        #Exergía destruida
        print("Exergía destruida")
        print("="*50)

        print(f"{'Evaporador LT [kW]:':25s} {self.Ex_d_ev/1000:.3f}")
        print(f"{'Condensador HT [kW]:':25s} {self.Ex_d_cond/1000:.3f}")
        print(f"{'HX cascada [kW]:':25s} {self.Ex_d_HX/1000:.3f}")
        print(f"{'Compresor LT [kW]:':25s} {self.Ex_d_comp_l/1000:.3f}")
        print(f"{'Compresor HT [kW]:':25s} {self.Ex_d_comp_h/1000:.3f}")
        print(f"{'Válvula LT [kW]:':25s} {self.Ex_d_valve_l/1000:.3f}")
        print(f"{'Válvula HT [kW]:':25s} {self.Ex_d_valve_h/1000:.3f}")
        print(f"{'Total [kW]:':25s} {self.Ex_d_total/1000:.3f}")

        