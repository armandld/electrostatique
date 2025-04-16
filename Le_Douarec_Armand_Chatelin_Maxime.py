import numpy as np
import subprocess
import matplotlib.pyplot as plt
import pdb
import os

# Parameters
# TODO adapt to what you need (folder path executable input filename)
executable = 'Exercice4'  # Name of the executable (NB: .exe extension is required on Windows)
repertoire = r"/Users/Sayu/Desktop/electrostatique"
os.chdir(repertoire)


CONFIG_FILE = os.path.join(os.path.dirname(__file__), "configuration.in.example")

input_filename = 'configuration.in.example'  # Name of the input file

def lire_configuration():
    config_path = os.path.join(os.path.dirname(__file__), "configuration.in.example")
    configuration = {}
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Le fichier {config_path} n'existe pas.")
    
    with open(config_path, "r", encoding="utf-8") as fichier:
        for ligne in fichier:
            ligne = ligne.strip()
            if ligne and "=" in ligne and not ligne.startswith("#"):
                cle, valeur = ligne.split("=", 1)
                configuration[cle.strip()] = valeur.strip()
    
    return configuration

def ecrire_configuration(nouvelles_valeurs):
    """Écrit les nouvelles valeurs dans le fichier de configuration."""
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"Le fichier {CONFIG_FILE} n'existe pas.")

    lignes_modifiees = []
    
    with open(CONFIG_FILE, "r", encoding="utf-8") as fichier:
        for ligne in fichier:
            ligne_strippée = ligne.strip()
            if ligne_strippée and "=" in ligne_strippée and not ligne_strippée.startswith("#"):
                cle, _ = ligne_strippée.split("=", 1)
                cle = cle.strip()
                if cle in nouvelles_valeurs:
                    ligne = f"{cle} = {nouvelles_valeurs[cle]}\n"
            lignes_modifiees.append(ligne)

    with open(CONFIG_FILE, "w", encoding="utf-8") as fichier:
        fichier.writelines(lignes_modifiees)

R=0.05
r1=0.025
epsilon_a = 1
epsilon_b = 1
uniform_rho_case = 1
VR = 10
N1=1000
N2=1000
verbose=1 
rho_0=1e4
epsilon_0=1

valeurs = lire_configuration()

def actualise_valeur():
    global R,r1,epsilon_a,epsilon_b,uniform_rho_case,VR,N1,N2,verbose
    R = float(valeurs.get("R"))
    epsilon_a = float(valeurs.get("epsilon_a"))
    epsilon_b = float(valeurs.get("epsilon_b"))
    uniform_rho_case = float(valeurs.get("uniform_rho_case"))
    VR = float(valeurs.get("VR"))
    N1 = float(valeurs.get("N1"))
    N2 = float(valeurs.get("N2"))
    verbose = float(valeurs.get("verbose"))

def ecrire_valeur(nom,valeur):
    global valeurs
    valeurs[nom] = valeur
    ecrire_configuration(valeurs)
    actualise_valeur()

def lancer_simulation(theta0, output_file):
    ecrire_configuration({"theta0": theta0})
    cmd = f"./{executable} {input_filename} output={output_file}"
    subprocess.run(cmd, shell=True)

outputs = []  # Liste pour stocker les fichiers de sortie
errors = []  # Liste pour stocker les erreurs
values = []
NS = [10, 100, 1000, 2000, 10000, 30000, 100000]  # Nombre de pas par période


paramstr = 'N1'  # Paramètre à scanner
param = NS

# Question 1

'''ARMANDDDDDDDDDDDD
for i, N in enumerate(param):
    ecrire_valeur("N2",N)
    output_file = f"{paramstr}={N}"
    outputs.append(output_file)
    cmd = f"./{executable} {input_filename} {paramstr}={N} output={output_file}"
    print(cmd)
    subprocess.run(cmd, shell=True)
    print('Simulation terminée.')

    # Chargement des données
    data = np.loadtxt(output_file+"_phi.txt")
    r = data[:, 0]
    phi = data[:, 1]
    values.append(phi[0]- 0.000625)
    # Solution analytique
    
    # Calcul de l'erreur à tfin
    #errors.append(delta)

# Tracé de l'étude de convergence
NS = np.array(NS)
plt.figure()
plt.loglog(NS, values, marker='v',markersize = 3, color = "black", linestyle='-')
#plt.plot(NS**2, values, marker='v',markersize = 3, color = "black", linestyle='-')
plt.ylabel("$\\phi(0)$ [rad]")
plt.xlabel("N")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.title("Convergence de $\\phi(0)$ en fonction de $N$")


# Question 2

N1S = [10, 100, 200, 500, 1000, 2000, 5000, 10000]  # Nombre de pas par période

plt.figure()

for i, N1k in enumerate(N1S):
        output_file = f"{'N1'}={N1k}{'N2'}={N1k}"
        outputs.append(output_file)
        cmd = f"./{executable} {input_filename} {'N1'}={N1k} {'N2'}={N1k} output={output_file}"
        print(cmd)
        subprocess.run(cmd, shell=True)
        print('Simulation terminée.')

        # Chargement des données
        data_phi = np.loadtxt(output_file+"_phi.txt")
        data_E = np.loadtxt(output_file+"_E.txt")
        data_D = np.loadtxt(output_file+"_D.txt")
        r = data_phi[1:, 0]
        phi = data_phi[1:, 1]
        E = data_E[:,1]
        D = data_D[:,1]
        plt.plot(r, E, marker='v',markersize = 3, linestyle='-', label = f"{'N1'}={N1k}{'N2'}={N1k}")


#plt.plot(NS**2, values, marker='v',markersize = 3, color = "black", linestyle='-')
plt.ylabel("$\\phi$ [rad]")
plt.xlabel("$r$ [m]")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.title("Convergence de $\\phi(0)$ en fonction de $N$")


plt.show()
'''





''' MAXIMEEEEEEEEEEEEEEEEEEEEEE

phi0_exact = (1 / 4) * R**2  # 0.000625 V

# Question 1 : étude de convergence de phi(0)
NS = [20, 100, 200, 500, 1000, 2000, 5000, 10000]
values = []

for N in NS:
    output_file = f"{'N1'}={N}{'N2'}={N}"
    outputs.append(output_file)
    cmd = f"./{executable} {input_filename} {'N1'}={N} {'N2'}={N} output={output_file}"
    print(cmd)
    subprocess.run(cmd, shell=True)
    print('Simulation terminée.')
    
    data = np.loadtxt(output_file + "_phi.txt")
    phi = data[:, 1]
    values.append(abs(phi[0] - phi0_exact))

plt.figure()
plt.loglog(NS, values, marker='v', markersize=3, color="black", linestyle='-')
plt.ylabel("Erreur sur $\\phi(0)$ [V]")
plt.xlabel("N")
plt.grid(True, linestyle="--", alpha=0.3)
plt.title("Convergence de $\\phi(0)$ en fonction de $N$")

# Question 2 : comparaison avec la solution analytique
'''
def phi_analytique(r):
    return (1 / 4) * (R**2 - r**2)

N1S = [100, 200, 400, 700, 2000, 10000]

'''
plt.figure()

for N1k in N1S:
    output_file = f"{'N1'}={N1k}{'N2'}={N1k}"
    outputs.append(output_file)
    cmd = f"./{executable} {input_filename} {'N1'}={N1k} {'N2'}={N1k} output={output_file}"
    print(cmd)
    subprocess.run(cmd, shell=True)
    ('Simulation terminée.')
    output_file = f"N1={N1k}N2={N1k}"
    data_phi = np.loadtxt(output_file + "_phi.txt")
    r = data_phi[1:, 0]  # On ignore r=0 pour comparaison
    phi_num = data_phi[1:, 1]
    phi_ana = phi_analytique(r)
    erreur = phi_num - phi_ana
    plt.plot(r, erreur, linestyle='--', marker='o', markersize=3, label=f"Erreur N={N1k}")

plt.xlabel("$r$ [m]")
plt.ylabel("Erreur $\\phi_{num} - \\phi_{ana}$ [V]")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.title("Différence entre potentiel numérique et analytique")


plt.plot(r, data_phi[1:, 1], linestyle='--', marker='o', markersize=3, label=f"Erreur N={N1k}")
plt.xlabel("$r$ [m]")
plt.ylabel("$\\phi_{num} [V]")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.title("Plot de $\\phi_{num}$ en fonction du rayon")
'''
for N1 in N1S:
    N2 = N1
    output_file = f"N1={N1}N2={N2}"
    cmd = f"./{executable} {input_filename} N1={N1} N2={N2} output={output_file}"
    print(cmd)
    subprocess.run(cmd, shell=True)
    data_phi = np.loadtxt(output_file + "_phi.txt")
    r = data_phi[1:, 0]  # On ignore r=0 pour comparaison
    phi_num = data_phi[1:, 1]
    phi_ana = phi_analytique(r)
    erreur = phi_num - phi_ana
    plt.plot(r, erreur, linestyle='-', label=f"Erreur pour N={N1}")

plt.xlabel("$r$ [m]")
plt.ylabel("Erreur $\\phi_{num} - \\phi_{ana}$ [V]")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.title("Différence entre potentiel numérique et analytique")

'''
plt.xlabel("$r$ [m]")
plt.ylabel("$\\phi(r)$ [V]")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.title("Potentiel électrique $\\phi_{num}$ en fonction du rayon")

plt.show()


plt.show()


plt.figure()
phi_r1_vals = []
inv_N2_squared = []

N1_list = [10000, 2500, 1000, 500, 300, 230, 175, 130, 100]


'''

'''
for N1 in N1_list:
    N2 = int(N1 * R / r1)
    output_file = f"{'N1'}={N1}{'N2'}={N2}"
    outputs.append(output_file)
    cmd = f"./{executable} {input_filename} {'N1'}={N1} {'N2'}={N2} output={output_file}"
    print(cmd)
    subprocess.run(cmd, shell=True)
    ('Simulation terminée.')
    data_phi = np.loadtxt(output_file + "_phi.txt")
    r = data_phi[:, 0]
    phi = data_phi[:, 1]
    idx_r1 = np.argmin(np.abs(r - r1))
    phi_r1_vals.append(phi[idx_r1])
    inv_N2_squared.append(1.0 / N2**2)

plt.plot(inv_N2_squared, phi_r1_vals, marker='s', linestyle='-', color='blue')
plt.xlabel("$1/N_2^2$")
plt.ylabel("$\\phi(r_1)$ [V]")
plt.grid(True, linestyle="--", alpha=0.3)
plt.title("Convergence quadratique de $\\phi(r_1)$ avec $N_2 ∝ N_1$")


plt.figure()

''' 
x = [10, 100, 1000, 10000]

plt.figure()
for N1 in x:
    N2 = N1
    output_file = f"N1={N1}N2={N2}"
    cmd = f"./{executable} {input_filename} N1={N1} N2={N2} output={output_file}"
    print(cmd)
    subprocess.run(cmd, shell=True)
    data_phi = np.loadtxt(output_file + "_phi.txt")
    r = data_phi[:, 0]
    phi = data_phi[:, 1]
    plt.plot(r, phi, linestyle='-', label=f"$N_1={N1}$")

plt.xlabel("$r$ [m]")
plt.ylabel("$\\phi(r)$ [V]")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.title("Potentiel électrique $\\phi(r)$")

plt.figure()
for N1 in x:
    N2 = N1
    output_file = f"N1={N1}N2={N2}"
    cmd = f"./{executable} {input_filename} N1={N1} N2={N2} output={output_file}"
    print(cmd)
    subprocess.run(cmd, shell=True)
    data_E = np.loadtxt(output_file + "_E.txt")
    r = data_E[:, 0]
    Er = data_E[:, 1]
    plt.plot(r, Er, linestyle='-', label=f"$N_1={N1}$")

plt.xlabel("$r$ [m]")
plt.ylabel("$E_r(r)$ [V/m]")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.title("Champ électrique radial $E_r(r)$")

plt.figure()
for N1 in x:
    N2 = N1
    output_file = f"N1={N1}N2={N2}"
    cmd = f"./{executable} {input_filename} N1={N1} N2={N2} output={output_file}"
    print(cmd)
    subprocess.run(cmd, shell=True)
    data_D = np.loadtxt(output_file + "_D.txt")
    r = data_D[:, 0]
    Dr = data_D[:, 1]
    plt.plot(r, Dr, linestyle='-', label=f"$N_1={N1}$")

plt.xlabel("$r$ [m]")
plt.ylabel("$D_r(r)$ [C/m^2]")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.title("Champ de déplacement $D_r(r)$")

'''
ecrire_valeur(r1, 0.015)

epsilon_b_values = [1, 10, 100, 1000, 10000]

    
for eps in epsilon_b_values:
    for N1 in [1000]:
        N2 = int(N1 * R / r1)
        output_file = f"N1={N1}N2={N2}_eps={eps}"
        outputs.append(output_file)
        cmd = f"./{executable} {input_filename} N1={N1} N2={N2} epsilon_b={eps} output={output_file}"
        print(cmd)
        subprocess.run(cmd, shell=True)
        data_E = np.loadtxt(output_file + "_phi.txt")
        r = data_E[:, 0]
        Er = data_E[:, 1]
        plt.plot(r, Er, linestyle='-', label=f"$\\phi$, $\\alpha={eps}$")

plt.xlabel("$r$ [m]")
plt.ylabel("$\phi(r)$ [V]")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.title("Potentiel électrique $\phi(r)$ pour divers $\\alpha$")

plt.figure()

for eps in epsilon_b_values:
    for N1 in [1000]:
        N2 = int(N1 * R / r1)
        output_file = f"N1={N1}N2={N2}_eps={eps}"
        outputs.append(output_file)
        cmd = f"./{executable} {input_filename} N1={N1} N2={N2} epsilon_b={eps} output={output_file}"
        print(cmd)
        subprocess.run(cmd, shell=True)
        data_E = np.loadtxt(output_file + "_E.txt")
        r = data_E[:, 0]
        Er = data_E[:, 1]
        plt.plot(r, Er, linestyle='-', label=f"$E_r$, $\\alpha={eps}$")

plt.xlabel("$r$ [m]")
plt.ylabel("$E_r(r)$ [V/m]")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.title("Champ électrique radial $E_r(r)$ pour divers $\\alpha$")

plt.figure()
for eps in epsilon_b_values:
    for N1 in [1000]:
        N2 = int(N1 * R / r1)
        output_file = f"N1={N1}N2={N2}_eps={eps}"
        outputs.append(output_file)
        cmd = f"./{executable} {input_filename} N1={N1} N2={N2} epsilon_b={eps} output={output_file}"
        print(cmd)
        subprocess.run(cmd, shell=True)
        data_D = np.loadtxt(output_file + "_D.txt")
        r = data_D[:, 0]
        D = data_D[:, 1]
        plt.plot(r, D, linestyle='-', label=f"$D_r$, $\\alpha={eps}$")

plt.xlabel("$r$ [m]")
plt.ylabel("$D_r(r)$ [C/m^2]")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.title("Champ de déplacement $D_r(r)$ pour divers $\\alpha$")

plt.show()


'''
plt.figure()
k1 = np.linspace(0,0.015,1000)
k2 = np.linspace(0.015,0.05,1000)
t1 = (1e4)*np.sin((np.pi)*k1/0.015)
t2 = k2*0
for N1 in [1000]:
    N2 = N1
    output_file = f"N1={N1}N2={N2}"
    data_D = np.loadtxt(output_file + "_D.txt")
    r = data_D[:, 0]
    D = data_D[:, 1]

    # Approximation de div(D) = (1/r) d(r D)/dr
    dr = r[1] - r[0]
    r_mid = (r[1:] + r[:-1]) / 2
    d_rD_dr = (r[1:] * D[1:] - r[:-1] * D[:-1]) / dr
    div_D = d_rD_dr / r_mid

    #plt.plot(r_mid, div_D, label=f"div(D) pour N={N1}")

    # Charge totale via div(D)
    rho_num = div_D
    Q_num = 2 * np.pi * np.trapezoid(rho_num * r_mid, r_mid)

    # Charge libre : ici rho_lib = epsilon_0
    Q_lib = np.pi * R**2 * epsilon_0

    # Charge de polarisation sur r = r1
    idx_r1 = np.argmin(np.abs(r - r1))
    D_r1 = D[idx_r1]
    Q_pol = 2 * np.pi * r1 * D_r1

    print(f"N={N1} : Q_num = {Q_num:.3e}, Q_lib = {Q_lib:.3e}, Q_pol = {Q_pol:.3e}")

plt.plot(k1,t1,linestyle = "-",color = "red")
plt.plot(k2,t2,linestyle = "-",color = "red", label=f"div(D) pour N={N1}")
plt.plot(k1,t1,linestyle = "--",color = "blue")
plt.plot(k2,t2,linestyle = "--",color = "blue", label="$\\rho_{lib}(r)$")
plt.xlabel("$r$ [m]")
plt.ylabel("$\\nabla$ $\cdot$ $D(r)$")
plt.title("Vérification de $\\nabla$ $\cdot$ $D(r)=ρ_{lib}(r)$")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()

plt.show()


k1 = np.linspace(0,0.015,1000)
k2 = np.linspace(0.015,0.05,1000)
t1 = (1e4)*np.sin((np.pi)*k1/0.015)
t2 = k2*0
