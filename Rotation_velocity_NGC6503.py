#Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit

#Data for fit
velocity_rot_data = np.array([77.0, 100.0, 110.0, 118.0, 122.0, 122.0, 118.0, 117.0, 117.0, 117.0, 115.0, 115.0, 116.0, 117.0, 117.0, 119.0, 118.0, 117.0, 115.0, 118.0, 117.0, 115.0, 115.0, 115.0, 113.0, 115.0, 113.0, 117.0, 115.0, 118.0, 115.0]) #Rotation velocity in km/s
radius_data = np.array([0.7, 1.4, 2.1, 2.8, 3.5, 4.2, 4.9, 5.6, 6.3, 7.0, 7.7, 8.4, 9.1, 9.8, 10.5, 11.2, 11.9, 12.6, 13.3, 14.0, 14.7, 15.4, 16.1, 16.8, 17.5, 18.2, 18.9, 19.6, 20.3, 21.0, 21.7]) #Radius in kpc

#Show the data points in a plot
font1={'size':16}
font2={'family':'serif','size':20}
plt.figure(figsize=(6,6))
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel('Radius [kpc]', fontdict=font1)
plt.ylabel('Velocity [km s$^{-1}]$', fontdict=font1)
plt.title('Rotacional velocity of NGC 6403', fontdict=font2)
plt.scatter(radius_data, velocity_rot_data, marker = '.', color = 'black')
plt.xlim(0,30)
plt.ylim(0,155)
plt.savefig('data.jpg')
print("Saved data plot")

#Fitting the rotational speed
#Parameters
radius_NGC = 9.198 #kpc units
Gravitational_constant = 6.6743e-11 #m**3/kg*s**2 units
factor_conversion = 2.9389e58 #m**3 equivalent to 1 kpc**3
G = Gravitational_constant/factor_conversion #kpc**3/kg*s**2 Gravitational constant
factor_conversion2 = 3.086e19 #km equivalent to 1 kpc
print("Radius of galaxy NGC6503:", radius_NGC, "kpc")
print("Garvitational constan:", Gravitational_constant, "m**3/kg*s**2")

#Definitions of velocity functions
def Integral_NFW(r, r0):
    return r/(r0 + r)**2

def Integral_SFDM(r, r0):
    return (np.sin(r/r0))**2

def velocity_NFW(r_d, r0, rho0, v, b):
    Mass_Integral, err = quad(Integral_NFW, 0, radius_NGC, args = (r0))
    vel_DM_NFW = (4 * np.pi * G * rho0 * r0**3 * Mass_Integral)/r_d
    vel_L = (v**2 * b * r0**2.21 * 1.97 * r_d**1.22)/(r_d**2 + 0.78**2 * r0**2)**1.43
    return  np.sqrt(vel_DM_NFW + vel_L) * factor_conversion2

def velocity_SFDM(r_d, r0, rho0, v, b):
    Mass_Integral, err = quad(Integral_SFDM, 0, radius_NGC, args = (r0))
    vel_DM_SFDM = (4 * np.pi * G * rho0 * r0**2 * Mass_Integral)/r_d
    vel_L = (v**2 * b * r0**2.21 * 1.97 * r_d**1.22)/(r_d**2 + 0.78**2 * r0**2)**1.43
    return  np.sqrt(vel_DM_SFDM + vel_L) * factor_conversion2

#Determining the parameters: r_0, rho_0, v_Ropt, beta, to fitting the rotational speed curve
popt_NFW, _ = curve_fit(velocity_NFW, radius_data, velocity_rot_data, p0=[14, 46906970, 11, 25], maxfev=100000)
radius_0_NFW, rho_0_NFW, v2_Ropt_NFW, beta_NFW = popt_NFW

popt_SFDM, _ = curve_fit(velocity_SFDM, radius_data, velocity_rot_data, p0=[14.163035224638316, 46906970.58375084, 11.047976404352129, 25.40437984698301], maxfev=100000)
radius_0_SFDM, rho_0_SFDM, v2_Ropt_SFDM, beta_SFDM = popt_SFDM

#New list of radius data to plotting the fit
radius_list = np.linspace(0.5, 21.7, 1000)

#Obtaining velocities from NFW and SFDM models to plotting the rotational curve fit
vel_NFW = list()
for i in radius_list:
    vel = velocity_NFW(i, radius_0_NFW, rho_0_NFW, v2_Ropt_NFW, beta_NFW)
    vel_NFW.append(vel)
vel_NFW = np.array(vel_NFW)

vel_SFDM = list()
for j in radius_list:
    vel2 = velocity_SFDM(j, radius_0_SFDM, rho_0_SFDM, v2_Ropt_SFDM, beta_SFDM)
    vel_SFDM.append(vel2)
vel_SFDM = np.array(vel_SFDM)

plt.figure(figsize=(6,6))
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel('Radius [kpc]', fontdict=font1)
plt.ylabel('Velocity [km s$^{-1}]$', fontdict=font1)
plt.title('Rotacional velocity of NGC 6403', fontdict=font2)
plt.scatter(radius_data, velocity_rot_data, marker = '.', color = 'black')
plt.plot(radius_list, vel_NFW, color = 'b', linestyle='dashed')
plt.plot(radius_list, vel_SFDM, color = 'red')
plt.legend(['Data', 'NFW Model', 'SFDM Model'], loc='best')
plt.xlim(0,30)
plt.ylim(0,155)
plt.savefig('fitting_rotational_velocity.jpg')
print("Saved fitting rotational velocity plot")

print("Fit parameters for NFW model")
print("Radius of halo core:", round(radius_0_NFW, 2), "kpc")
print("Density of halo core:", round(rho_0_NFW, 2), "kg/kpc**3")

print("Fit parameters for SFDM model")
print("Radius of halo core:", round(radius_0_SFDM, 2), "kpc")
print("Density of halo core:", round(rho_0_SFDM, 2), "kg/kpc**3")