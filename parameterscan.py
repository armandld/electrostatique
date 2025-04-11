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
r1=0.015
epsilon_a = 1
epsilon_b = 1
uniform_rho_case = 1
VR = 0
N1=10
N2=10
verbose=1 


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
NS = [1000, 2000, 5000, 7000, 10000]  # Nombre de pas par période


paramstr = 'N1'  # Paramètre à scanner
param = NS

# Question 1

ecrire_valeur("r1",0.015)
ecrire_valeur("R",0.05)
ecrire_valeur("VR",0)

'''
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
'''

# Question 2

ecrire_valeur("epsilon_a",1)
ecrire_valeur("epsilon_b",4)
ecrire_valeur("pho0",1e4)
N1S = [100, 2000, 50000]  # Nombre de pas par période
N2S = [100, 2000, 50000]

plt.figure()

for i, N1k in enumerate(N1S):
    for j, N2k in enumerate(N2S):
        output_file = f"{'N1'}={N1k}{'N2'}={N2k}"
        outputs.append(output_file)
        cmd = f"./{executable} {input_filename} {'N1'}={N1k} {'N2'}={N2k} output={output_file}"
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
        plt.plot(r, E, marker='v',markersize = 3, linestyle='-', label = f"{'N1'}={N1k}{'N2'}={N2k}")


#plt.plot(NS**2, values, marker='v',markersize = 3, color = "black", linestyle='-')
plt.ylabel("$\\phi$ [rad]")
plt.xlabel("$r$ [m]")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.title("Convergence de $\\phi(0)$ en fonction de $N$")


plt.show()

