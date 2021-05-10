# copy of notebook but in Julia
using LinearAlgebra, SpecialFunctions, Printf

function xyz_reader(file_name)
    file = open(file_name)

    num_atoms = 0
    atom_type = []
    atom_coordinates = []

    lines = readlines(file)
    for (idx, line) in enumerate(lines)
        # get num atoms
        if idx==1
            try
                num_atoms = line
            catch
                println("xyz file not in correct format. Make sure the format follows=> https=>//en.wikipedia.org/wiki/XYZ_file_format")
            end
        end

        if idx==2
            continue
        end

        # get atom types and positions
        if idx != 1
            theSplit = split(line, " ")
            atom = theSplit[1]
            coordinates = [parse(Float64, theSplit[2]), parse(Float64, theSplit[3]), parse(Float64, theSplit[4])]
            push!(atom_type, atom)
            push!(atom_coordinates, coordinates)
        end
    end

    close(file)

    return parse(Int64, num_atoms), atom_type, atom_coordinates
end

file_name = "HeH.xyz"
const N_atoms, atoms, atom_coordinates = xyz_reader(file_name)

STOnG = 3
const zeta_dict = Dict("H"=> [1.24], "He"=>[2.0925], "Li"=>[2.69,0.80],"Be"=>[3.68,1.15], "B"=>[4.68,1.50],"C"=>[5.67,1.72])
# Dictionary containing the max quantum number of each atom,
# for a minimal basis STO-nG calculation
const max_quantum_number = Dict("H"=>1,"He"=>1,"Li"=>2,"Be"=>2,"C"=>2)

# Gaussian contraction coefficients (pp157)
# Going up to 2s orbital (W. J. Hehre, R. F. Stewart, and J. A. Pople. J. Chem. Phys. 51, 2657 (1969))
# Row represents 1s, 2s etc...
D = [[0.444635, 0.535328, 0.154329],[0.700115,0.399513,-0.0999672]]

# Gaussian orbital exponents (pp153)
# Going up to 2s orbital (W. J. Hehre, R. F. Stewart, and J. A. Pople. J. Chem. Phys. 51, 2657 (1969))
const alpha = [[0.109818, 0.405771, 2.22766],[0.0751386,0.231031,0.994203]]

# basis set size
Btmp = 0
for atom in atoms
    global Btmp
    Btmp += max_quantum_number[atom]
end
const B = Btmp

# num electrons
const N = 2

# dict of charges
const charge_dict = Dict("H"=> 1, "He"=> 2, "Li"=>3, "Be"=>4,"B"=>5,"C"=>6,"N"=>7,"O"=>8,"F"=>9,"Ne"=>10)

# calculate the integrals between Gaussian orbitals (pp410, appendix A)

function gauss_product(gauss_A, gauss_B)
    # The product of two Gaussians gives another Gaussian (pp411)
    # Pass in the exponent and centre as a tuple
    a, Ra = gauss_A
    b, Rb = gauss_B
    p = a + b
    diff = norm(Ra .- Rb)^2
    N = (4*a*b/(pi^2))^0.75 # normalization
    K = N*exp(-a*b/p*diff) # new prefactor
    Rp = (a.*Ra .+ b.*Rb)./p # new center
    return p, diff, K, Rp
end

# overlap integral
function overlap(A, B)
    p, diff, K, Rp = gauss_product(A,B)
    prefactor = (pi/p)^1.5
    return prefactor*K
end

# kinetic integral
function kinetic(A, B)
    p, diff, K, Rp = gauss_product(A, B)
    prefactor = (pi/p)^1.5

    a, Ra = A
    b, Rb = B
    reduced_exponent = a*b/p
    return reduced_exponent * (3 - 2*reduced_exponent*diff) * prefactor*K
end

# variant of erf
# used for potential and e-e repulsion integrals
function Fo(t)
    if t == 0
        return 1
    else
        return (0.5*(pi/t)^0.5)*erf(t^0.5)
    end
end

# nuclear-electron integral
function potential(A, B, atom_idx)
    p, diff, K, Rp = gauss_product(A,B)
    Rc = atom_coordinates[atom_idx]
    Zc = charge_dict[atoms[atom_idx]]

    return (-2*pi*Zc/p) * K * Fo(p*norm(Rp.-Rc)^2)
end

# (ab|cd) integral

function multi(A, B, C, D)
    p, diff_ab, K_ab, Rp = gauss_product(A,B)
    q, diff_cd, K_cd, Rq = gauss_product(C,D)
    multi_prefactor = 2*pi^2.5*(p*q*(p+q)^0.5)^-1
    return multi_prefactor*K_ab*K_cd*Fo(p*q/(p+q)*norm(Rp.-Rq)^2)
end

# initialize matrices
S = zeros(B,B)
T = zeros(B,B)
V = zeros(B,B)
multi_elec_tensor = zeros(B,B,B,B)

# iterate through atoms
for (idx_a, val_a) in enumerate(atoms)
    # for each atom, get charge and center
    Za = charge_dict[val_a]
    Ra = atom_coordinates[idx_a]

    # iterate through quantum numbers (1s, 2s, etc)
    for m = 1:max_quantum_number[val_a]
        # for each quantum number: get contraction coeffs, then zeta, then scale exponents
        d_vec_m = D[m]
        zeta = zeta_dict[val_a][m]
        alpha_vec_m = alpha[m].*zeta^2

        # iterate over the contraction coeffs
        for p = 1:STOnG
            # iterate through the atoms once again
            for (idx_b, val_b) in enumerate(atoms)
                Zb = charge_dict[val_b]
                Rb = atom_coordinates[idx_b]
                for n = 1:max_quantum_number[val_b]
                    d_vec_n = D[n]
                    zeta = zeta_dict[val_b][n]
                    alpha_vec_n = alpha[n].*zeta^2

                    for q in 1:STOnG
                        # verified that 1-based indexing solves it
                        a = idx_a
                        b = idx_b

                        # generate the overlap, kinetic, and potential matrices
                        S[a,b] += d_vec_m[p]*d_vec_n[q]*overlap((alpha_vec_m[p],Ra),(alpha_vec_n[q],Rb))
                        T[a,b] += d_vec_m[p]*d_vec_n[q]*kinetic((alpha_vec_m[p],Ra),(alpha_vec_n[q],Rb))

                        for i = 1:N_atoms
                            V[a,b] += d_vec_m[p]*d_vec_n[q]*potential((alpha_vec_m[p],Ra),(alpha_vec_n[q],Rb),i)
                        end

                        # 2 more iterations to get the multi-electron tensor
                        for (idx_c, val_c) in enumerate(atoms)
                            Zc = charge_dict[val_c]
                            Rc = atom_coordinates[idx_c]
                            for k = 1:max_quantum_number[val_c]
                                d_vec_k = D[k]
                                zeta = zeta_dict[val_c][k]
                                alpha_vec_k = alpha[k].*zeta^2
                                for r = 1:STOnG
                                    for (idx_d, val_d) in enumerate(atoms)
                                        Zd = charge_dict[val_d]
                                        Rd = atom_coordinates[idx_d]
                                        for l = 1:max_quantum_number[val_d]
                                            d_vec_l = D[l]
                                            zeta = zeta_dict[val_d][l]
                                            alpha_vec_l = alpha[l].*zeta^2

                                            for s = 1:STOnG
                                                c = idx_c
                                                d = idx_d
                                                (
                                                multi_elec_tensor[a,b,c,d] += d_vec_m[p]*d_vec_n[q]*d_vec_k[r]*d_vec_l[s]
                                                    * multi((alpha_vec_m[p],Ra),
                                                    (alpha_vec_n[q],Rb),
                                                    (alpha_vec_k[r],Rc),
                                                    (alpha_vec_l[s],Rd))
                                                )
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

# form Hcore
Hcore = T + V

function pretty(name::String, thing)
    println("\n--- ", name)
    show(stdout, "text/plain", thing)
    println("\n")
end

#symmetric orthog. of basis

# Julia apparently does things backward from the default numpy behavior
#   like the order of eigenvalues, but it's okay
evalS, U = eigen(S)
diagS = transpose(U) * (S * U)
diagS_minushalf = Diagonal((Diagonal(diagS)^-0.5))
X = U * (diagS_minushalf * transpose(U))

function SD_successive_density_matrix_elements(Ptilde, P)
    x = 0
    for i = 1:B
        for j = 1:B
            x += (B^-2) * (Ptilde[i,j] - P[i,j])^2
        end
    end
    return x^0.5
end

######## Algorithm ###########

# initial guess at P

P = zeros(B, B)
P_prev = zeros(B, B)
P_list = []

# interative process
threshold = 100
while threshold > 1e-4
    global threshold

    # calculate the Fock matrix with a guess
    G = zeros(B, B)
    for i = 1:B
        for j = 1:B
            for x = 1:B
                for y = 1:B
                    G[i,j] += P[x,y] * (multi_elec_tensor[i,j,y,x] - 0.5*multi_elec_tensor[i,x,y,j])
                end
            end
        end
    end
    Fock = Hcore + G

    # calculate the Fock matrix in orthogonalized base
    Fockprime = transpose(X) * (Fock * X)
    evalFockprime, Cprime = eigen(Fockprime)

    C = X * Cprime

    # form a new P
    # note: we only sum over the electron PAIRS, NOT the whole basis set
    for i = 1:B
        for j = 1:B
            for a = 1:convert(Int, N/2)
                P[i,j] = 2 * C[i,a] * C[j,a]
            end
        end
    end

    push!(P_list, P)

    threshold = SD_successive_density_matrix_elements(P_prev, P)

    copy!(P_prev, P)

    if threshold < 1e-4
        @printf "STO3G Restricted Closed Shell HF algorithm took %i iterations to converge\n\n" length(P_list)
        @printf "The orbital energies are %.10f and %.10f Hartrees\n\n" evalFockprime[1] evalFockprime[2]
        pretty("The orbital matrix is: ", C)
        pretty("The density/bond order matrix is: ", P)
    end
end

println("done!")