#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ConfigFile.tpp"

using namespace std;

const double PI=3.1415926535897932384626433832795028841971e0;

// Résolution d'un système d'équations linéaires par élimination de
// Gauss-Jordan:
template<class T>
vector<T>
solve(const vector<T>& diag,
      const vector<T>& lower,
      const vector<T>& upper,
      const vector<T>& rhs)
{
    vector<T> solution(diag.size());
    vector<T> new_diag(diag);
    vector<T> new_rhs(rhs);

    for (int i = 1; i < diag.size(); ++i) {
        T pivot = lower[i - 1] / new_diag[i - 1];
        new_diag[i] -= pivot * upper[i - 1];
        new_rhs[i] -= pivot * new_rhs[i - 1];
    }

    solution[diag.size() - 1] =
      new_rhs[diag.size() - 1] / new_diag[diag.size() - 1];

    for (int i(diag.size() - 2); i >= 0; --i)
        solution[i] = (new_rhs[i] - upper[i] * solution[i + 1]) / new_diag[i];

    return solution;
}

//TODO build the epsilon function
double epsilon(double r, double r1, double epsilon_a, double epsilon_b)
{
    return (r < r1) ? epsilon_a : epsilon_b;
}
const double epsilon0 = 1;

//TODO build the rho_epsilon function (rho_lib / epsilon_0)
double rho_epsilon(double r, double r1, double rho0, bool uniform_case)
{
    if (uniform_case)
        return rho0;  // partout
    else if (r < r1)
        return rho0 * sin(PI * r / r1);
    else
        return 0.0;
}

int main(int argc, char* argv[])
{
    

    // USAGE: Exercise4 [configuration-file] [<settings-to-overwrite> ...]

    // Read the default input
    string inputPath = "configuration.in.example";
    // Optionally override configuration file.
    if (argc > 1)
        inputPath = argv[1];

    ConfigFile configFile(inputPath);
    // Override settings
    for (int i = 2; i < argc; i++)
        configFile.process(argv[i]);

    // Set verbosity level. Set to 0 to reduce printouts in console.
    const int verbose = configFile.get<int>("verbose");
    configFile.setVerbosity(verbose);

    // Read geometrical inputs
    const double R  = configFile.get<double>("R");
    const double r1 = configFile.get<double>("r1");

    const double rho0 = configFile.get<double>("rho0");

    // For the analytical comparison
    const bool uniform_rho_case = configFile.get<bool>("uniform_rho_case");
        
    // Dielectric relative permittivity
    const double epsilon_a = configFile.get<double>("epsilon_a");
    const double epsilon_b = configFile.get<double>("epsilon_b");
    
    // Boundary conditions
    const double VR = configFile.get<double>("VR");
    
    // Discretization
    const int N1 = configFile.get<int>("N1");
    const int N2 = configFile.get<int>("N2");
    
    // Fichiers de sortie:
    string fichier = configFile.get<string>("output");
    string fichier_phi = fichier+"_phi.txt";
    string fichier_E   = fichier+"_E.txt";
    string fichier_D   = fichier+"_D.txt";

    // Create our finite elements
    const int pointCount = N1 + N2 + 1; // Number of finite elements
    const double h1 = (r1 - 0) / N1;
    const double h2 = (R  - r1) / N2;

    // Position of elements
    vector<double> r(pointCount);

    //TODO build the nodes vector r
    for (int i = 0; i <= N1; ++i)
        r[i] = r1 * i / N1;
    for (int i = 1; i <= N2; ++i)
        r[N1 + i] = r[N1] + (R - r1) * i / N2;     
    // Arrays initialization
    vector<double> h(pointCount-1); 	// Distance between grid points
    vector<double> midPoint(pointCount-1);  // Midpoint of each grid element
        
    // TODO build the h vector and midpoint vector

    for (int i = 0; i < N1 + N2; ++i)
        h[i] = r[i + 1] - r[i];

    for (int i = 0; i < N1 + N2; ++i)
        midPoint[i] = 0.5 * (r[i] + r[i + 1]);
 
    // Construct the matrix and right-hand side
    vector<double> diagonal(pointCount, 0.0);  // Diagonal
    vector<double> lower(pointCount - 1, 0.0); // Lower diagonal
    vector<double> upper(pointCount - 1, 0.0); // Upper diagonal
    vector<double> rhs(pointCount, 0.0);       // Right-hand-side
    
    // Loop over the intervals: add the contributions to matrix and rhs   
    for (int k = 0; k < pointCount-1; ++k) {
        
                   
        // TODO build the vectors diagonal, lower, upper, rhs
        double coeff = epsilon(midPoint[k], r1, epsilon_a, epsilon_b) * midPoint[k] / h[k];
        diagonal[k]     += coeff;
        diagonal[k + 1] += coeff;
        lower[k]         = -coeff;
        upper[k]         = -coeff;
    
        double rho = rho_epsilon(midPoint[k], r1, rho0, uniform_rho_case);
        double rhs_contrib = 0.5 * h[k] * rho * midPoint[k];
        rhs[k]     += rhs_contrib;
        rhs[k + 1] += rhs_contrib;
    }  

    // TODO boundary condition at r=R (modify the lines below)
    lower[lower.size() - 1]       = 0.0;
    diagonal[diagonal.size() - 1] = 1.0;
    upper[upper.size() - 1]       = 0.0;
    rhs[rhs.size() - 1]           = VR;

    // Solve the system of equations
    vector<double> phi = solve(diagonal, lower, upper, rhs);

    // Calculate electric field E and displacement vector D
    vector<double> E(pointCount - 1, 0);
    vector<double> D(pointCount - 1, 0);
    for (int i = 0; i < E.size(); ++i) {
        // TODO calculate E and D
        double dphi = phi[i + 1] - phi[i];
        E[i] = -dphi / h[i];
        D[i] = epsilon0* epsilon(midPoint[i], r1, epsilon_a, epsilon_b) * E[i];
    }

    // Export data
    {
        // Electric potential phi
        ofstream ofs(fichier_phi);
        ofs.precision(15);

        if (r.size() != phi.size())
            throw std::runtime_error("error when writing potential: r and "
                                     "phi does not have size");

        for (int i = 0; i < phi.size(); ++i) {
            ofs << r[i] << " " << phi[i] << endl;
        }
    }

    {
        // Electric field E
        ofstream ofs(fichier_E);
        ofs.precision(15);

        if (r.size() != (E.size() + 1))
            throw std::runtime_error("error when writing electric field: size of "
                                     "E should be 1 less than r");

        for (int i = 0; i < E.size(); ++i) {
            ofs << midPoint[i] << " " << E[i] << endl;
        }
    }
    {
        // Displacement field D
        ofstream ofs(fichier_D);
        ofs.precision(15);

        if (E.size() != D.size())
            throw std::runtime_error("error when writing displacement field: size of "
                                     "D should be equal to E");

        for (int i = 0; i < D.size(); ++i) {
            ofs << midPoint[i] << " " << D[i] << endl;
        }
    }
    // === Vérification console ===
    std::cout << "Vérification des points aux bornes :" << std::endl;
    std::cout << "r[0]    = " << r[0] << ", phi[0] = " << phi[0] << std::endl;
    std::cout << "r[end]  = " << r[r.size() - 1] << ", phi[end] = " << phi[phi.size() - 1] << std::endl;

    // Différence phi(0) - phi(R) pour info
    std::cout << "Différence phi(0) - phi(R) = " << phi[0] - phi[phi.size() - 1] << std::endl;

    return 0;
}
