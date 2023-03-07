from qiskit.algorithms import VQE, NumPyMinimumEigensolver, NumPyEigensolver #Algorithms

#Qiskit odds and ends
from qiskit.circuit.library import EfficientSU2, EvolvedOperatorAnsatz
from qiskit.algorithms.optimizers import COBYLA, SPSA, SLSQP, L_BFGS_B
from qiskit.opflow import Z2Symmetries, X, Y, Z, I, PauliSumOp, Gradient, NaturalGradient
from qiskit import IBMQ, BasicAer, Aer, transpile
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.utils.mitigation import CompleteMeasFitter #Measurement error mitigatioin
from qiskit.tools.visualization import circuit_drawer
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.algorithms.minimum_eigensolvers import VQE, AdaptVQE, MinimumEigensolverResult
from qiskit.primitives import Estimator
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.quantum_info import SparsePauliOp

#qiskit_nature
from qiskit_nature.second_q.drivers import PySCFDriver, MethodType
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.circuit.library import UCCSD, PUCCD, SUCCD, HartreeFock, CHC, VSCF
from qiskit_nature.second_q.operators.fermionic_op import FermionicOp
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer , FreezeCoreTransformer
from qiskit_nature.second_q.problems import ElectronicStructureProblem, EigenstateResult
from qiskit_nature.second_q.mappers import QubitConverter, ParityMapper, BravyiKitaevMapper, JordanWignerMapper
from qiskit_nature.second_q.algorithms.ground_state_solvers.minimum_eigensolver_factories.vqe_ucc_factory import VQEUCCFactory
from qiskit_nature.second_q.algorithms.ground_state_solvers.minimum_eigensolver_factories.numpy_minimum_eigensolver_factory import NumPyMinimumEigensolverFactory
from qiskit_nature.second_q.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit_nature.second_q.algorithms.excited_states_solvers.eigensolver_factories.numpy_eigensolver_factory import NumPyEigensolverFactory
from qiskit_nature.second_q.algorithms.excited_states_solvers import QEOM, ExcitedStatesEigensolver

#Runtime
from qiskit_ibm_runtime import (QiskitRuntimeService, Session,
                                Estimator as RuntimeEstimator)
from qiskit_ibm_runtime.options import Options, ResilienceOptions, SimulatorOptions, TranspilationOptions, ExecutionOptions

#PySCF
from functools import reduce
import scipy.linalg
from pyscf import scf
from pyscf import gto, dft
from pyscf import mcscf, fci
from functools import reduce
from pyscf.mcscf import avas, dmet_cas

#Python odds and ends
import matplotlib
import matplotlib.pyplot as plt
import pylab
import numpy as np
import os
import pyscf
from IPython.display import display, clear_output
import mapomatic as mm

from datetime import datetime

## Python program to store list to file using pickle module
import pickle

# write list to binary file
def write_list(a_list,filename):
    # store list in binary file so 'wb' mode
    with open(filename, 'wb') as fp:
        pickle.dump(a_list, fp)
        print('Done writing list into a binary file')
        
def write_dict(a_dict,filename):
    # store list in binary file so 'wb' mode
    with open(filename, 'wb') as fp:
        pickle.dump(a_dict, fp,protocol=pickle.HIGHEST_PROTOCOL)
        print('Done writing dict into a binary file')

# Read list to memory
def read(filename):
    # for reading also binary mode is important
    with open(filename, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list

IBMQ.load_account()
service = QiskitRuntimeService(channel='ibm_quantum')
#set Backends
#simulators
seed=42

def callback(eval_count, param, val,meta):  
    # Overwrites the same line when printing
    counts.append(eval_count)
    interim_info['counts'].append(eval_count)
    values.append(val)
    interim_info['values'].append(val)
    params.append(param)
    interim_info['params'].append(param)
    meta_dicts.append(meta)
    mean=np.mean(values)
    std=np.std(values)

    write_dict(interim_info,f'../VQE_results/MG+CO2_ibmq_montreal_4qubit_interim_info_{int(distances[0]*10)}')
    display("Evaluation: {}, Energy: {}, Mean: {}, STD: {}, Metadata: {}".format(eval_count, val,mean,std, meta))
    clear_output(wait=True)

def adapt_solver(distances,mapper,optimizer,freeze_core):
    
    dists=[]
    results=[]
    problems=[]
    hf_energies=[]
    initial_points=[]
    ops=[]
    ansatze=[]
    for dist in distances:
        #Driver
        driver,og_problem=make_driver(dist)
        #Qubit_Op
        qubit_op, problem, converter,hf_energy = make_qubit_op(dist,og_problem,mapper,freeze_core)
        ops.append(qubit_op)
        problems.append(problem)
        #Initial State
        init_state = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, converter)
        
        #ansatz
        ansatz = UCCSD(num_spatial_orbitals=problem.num_spatial_orbitals,num_particles=problem.num_particles,qubit_converter=converter,initial_state=init_state)

        operator_pool = []
        for op in ansatz.operators:
            sop = op.primitive
            for pauli, coeff in zip(sop.paulis, sop.coeffs):
                if sum(pauli.x & pauli.z) % 2 == 0:
                    continue
            operator_pool.append(PauliSumOp(coeff * SparsePauliOp(pauli)))

        ansatz = EvolvedOperatorAnsatz(
        operators=operator_pool,
        initial_state=init_state,
        )
        ansatze.append(ansatz)
        
        
        # Set initial parameters of the ansatz
        if len(initial_points)!=0 and ops[-1].num_qubits==ops[-2].num_qubits:
            initial_point=initial_points[-1]
        elif len(initial_points)!=0:
            old_ans=ansatz[-1]
            if ansatz.num_parameters>old_ans.num_parameters:
                initial_point=np.append(initial_points[-1],np.zeros(ansatz.num_parameters - old_ans.num_parameters))
            else:
                to_remove=old_ans.num_parameters-ansatz.num_parameters
                initial_point=np.delete(initial_points[-1],np.arange(-1,-to_remove-1,-1))
        else:
            #initial_point = np.pi/4 * np.random.rand(ansatz.num_parameters)
            initial_point= np.zeros(ansatz.num_parameters)
        
        estimator = Estimator([ansatz], [qubit_op])
    
        counts = []
        values = []
        deviation = []
    
        custom_vqe =VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer,initial_point=initial_point, callback=callback)
        
        adapt_vqe=AdaptVQE(custom_vqe)
        adapt_vqe.supports_aux_operators = lambda: True
        adapt_vqe.threshold=1e-3
        adapt_vqe.max_iterations=4
        adapt_vqe.delta=1e-4
        
        solver = GroundStateEigensolver(converter, adapt_vqe)
        result = solver.solve(problem)
        
        initial_points.append(result.raw_result.optimal_point)
        results.append(result)
        dists.append(dist)
        hf_energies.append(hf_energy)
    
    return results, problems ,distances, hf_energies


def real_solver(distances, mapper, optimizer,freeze_core,est_options, device):
    
    dists=[]
    results=[]
    problems=[]
    hf_energies=[]
    initial_points=[]
    ops=[]
    ansatze=[]
                             
    for dist in distances:
        #Driver
        driver,og_problem=make_driver(dist)
        #Qubit_Op
        qubit_op, problem, converter,hf_energy = make_qubit_op(dist,og_problem,mapper,freeze_core)
        ops.append(qubit_op)
        problems.append(problem)
        #Initial State
        init_state = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, converter)
        
        #ansatz
        #ansatz = UCCSD(num_spatial_orbitals,num_particles,converter)
        
        ansatz = UCCSD(num_spatial_orbitals=problem.num_spatial_orbitals,num_particles=problem.num_particles,qubit_converter=converter,initial_state=init_state)

        operator_pool = []
        for op in ansatz.operators:
            sop = op.primitive
            for pauli, coeff in zip(sop.paulis, sop.coeffs):
                if sum(pauli.x & pauli.z) % 2 == 0:
                    continue
            operator_pool.append(PauliSumOp(coeff * SparsePauliOp(pauli)))

        ansatz = EvolvedOperatorAnsatz(
        operators=operator_pool,
        initial_state=init_state,
        )
        ansatze.append(ansatz)
        #ansatz = EfficientSU2(num_qubits=qubit_op.num_qubits,su2_gates='ry', entanglement='linear', reps=3, initial_state=init_state)
        
        #ansatz_opt = transpile(ansatz, backend=provider.get_backend(device),optimization_level=3,routing_method='sabre')
        #small_qc = mm.deflate_circuit(ansatz_opt)
        #layouts = mm.matching_layouts(small_qc, provider.get_backend(device))
        #scores = mm.evaluate_layouts(small_qc, layouts, provider.get_backend(device))
        #ansatz = transpile(small_qc, backend=provider.get_backend(device),initial_layout=scores[0][0],optimization_level=3,routing_method='sabre')
        
        # Set initial parameters of the ansatz
        if len(initial_points)!=0 and ops[-1].num_qubits==ops[-2].num_qubits:
            initial_point=initial_points[-1]
        elif len(initial_points)!=0:
            old_ans=ansatze[-1]
            if ansatz.num_parameters>old_ans.num_parameters:
                initial_point=np.append(initial_points[-1],np.zeros(ansatz.num_parameters - old_ans.num_parameters))
            else:
                to_remove=old_ans.num_parameters-ansatz.num_parameters
                initial_point=np.delete(initial_points[-1],np.arange(-1,-to_remove-1,-1))
        else:
            #initial_point = np.pi/4 * np.random.rand(ansatz.num_parameters)
            initial_point= np.zeros(ansatz.num_parameters)
    
        counts= []
        values = []
        deviation = []
        params=[]
    
        with Session(service=service, backend=device, max_time=288000) as session:
            # Prepare primitive
            rt_estimator = RuntimeEstimator(session=session,options=est_options)
            # Set up algorithm
            custom_vqe = VQE(rt_estimator, ansatz, optimizer,initial_point=initial_point, callback=callback)
            adapt_vqe=AdaptVQE(custom_vqe)
            adapt_vqe.supports_aux_operators = lambda: True
            adapt_vqe.threshold=1e-3
            adapt_vqe.max_iterations=4
            adapt_vqe.delta=1e-4
            # Run algorithm
            solver = GroundStateEigensolver(converter, adapt_vqe)
            result = solver.solve(problem)
            #result = custom_vqe.compute_minimum_eigenvalue(qubit_op,initial_point)
            
            
        initial_points.append(result.raw_result.optimal_point)
        results.append(result)
        dists.append(dist)
        hf_energies.append(hf_energy)
        
        problems.append(problem)
       
    return results,problems, distances

def make_driver(d):
    
    
    molecule = MoleculeInfo(
             # coordinates in Angstrom
                     symbols=['O','C','O','Mg'],
                     coords=[
                            (-1.1621,0.0,0.0),
                            (0.0,0.0,0),
                            (1.1621,0.0,0),
                            (d+1.1621, 0.0,0.0),
                            ],
                     multiplicity=1,  # = 2*spin + 1
                     charge=2,
                     units=DistanceUnit.ANGSTROM
                    )
    
    #Set driver
    #driver = PySCFDriver.from_molecule(molecule, basis="sto3g", method=MethodType.ROHF)
    #driver.xc_functional='pbe,pbe'
    driver = PySCFDriver.from_molecule(molecule, basis="6-31g*", method=MethodType.ROKS)
    driver.xc_functional='b3lyp'
    driver.conv_tol = 1e-6

    #Get properties
    problem = driver.run()
    

    return driver, problem

def make_qubit_op(d,og_problem, mapper,freeze_core):
    mol = gto.Mole()
    mol.atom = [
        ['C',(0.0,0.0,0)],
        ['O',(-1.1621,0.0,0.0),],
        ['O',(1.1621,0.0,0.0)],
        ['Mg',(d+1.1621, 0.0, 0.0)]
        ]
    mol.charge=2
    mol.basis = '6-31g*'
    mol.spin = 0
    mol.build()
    
    #mf= scf.ROHF(mol).x2c()
    mf = dft.ROKS(mol).density_fit(auxbasis='def2-universal-jfit')
    mf.xc ='pbe,pbe'
    mf.max_cycle = 50
    mf.conv_tol = 1e-6
    
    first_run=mf.kernel()
    a = mf.stability()[0]
    if(mf.converged):
        energy=first_run
    else:
        mf.max_cycle = 80
        mf.conv_tol = 1e-6
        mf = scf.newton(mf)
        scnd_run=mf.kernel(dm0 = mf.make_rdm1(a,mf.mo_occ)) # using rdm1 constructed from stability analysis
      #mf.kernel(mf.make_rdm1()) #using the rdm from the non-converged calculation
        if(mf.converged):
            energy=scnd_run
        else:
            mf.conv_tol = 1e-6
            mf.max_cycle = 80
            mf = scf.newton(mf) #Second order solver
            energy=mf.kernel(dm0 = mf.make_rdm1())


    ao_labels = ['Mg 2p', 'O 2p']
    avas_obj = avas.AVAS(mf, ao_labels)
    avas_obj.kernel()
    weights=np.append(avas_obj.occ_weights,avas_obj.vir_weights)
    weights=(weights>0.2)*weights
    orbs=np.nonzero(weights)
    orbs=np.nonzero(weights)
    
    transformer = ActiveSpaceTransformer(
            num_electrons=(3,3), #Electrons in active space
            num_spatial_orbitals=4, #Orbitals in active space
            #active_orbitals=orbs[0].tolist().append(orbs[0][-1]+1)
        )
    fz_transformer=FreezeCoreTransformer(freeze_core=freeze_core)
    
    #Define the problem

    problem=transformer.transform(og_problem)
    if freeze_core==True:
        problem=fz_transformer.transform(problem)
        converter = QubitConverter(mapper)
    else:
        converter = QubitConverter(mapper,two_qubit_reduction=True, z2symmetry_reduction='auto')

    hamiltonian=problem.hamiltonian
    second_q_op = hamiltonian.second_q_op()
    
    num_spatial_orbitals = problem.num_spatial_orbitals
    num_particles = problem.num_particles
    
    qubit_op = converter.convert(second_q_op,num_particles=num_particles,sector_locator=problem.symmetry_sector_locator)
    
        

    
    return qubit_op, problem,  converter, energy
    
    
#Runtime Estimator options
ts_opt=TranspilationOptions(
    skip_transpilation=False
)

res_opt=ResilienceOptions(
    noise_factors=tuple(range(1, 6, 2)),
    noise_amplifier='LocalFoldingAmplifier',
    extrapolator='LinearExtrapolator'
)

ex_opt=ExecutionOptions(
    shots=1024
)
est_options=Options(
    resilience_level=1,
    optimization_level=3,
    execution=ex_opt,
    #resilience=res_opt,
    #transpilation=ts_opt
)


distances = [0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3.0,3.3]
optimizer = SPSA(maxiter=100)
mapper=ParityMapper()

algorithm_globals.random_seed = seed

counts = []
values = []
params = []
meta_dicts=[]
interim_info={'counts':[],
              'values':[],
              'params':[]
             }


vqe_results,vqe_problems,dists=real_solver(distances=[distances[0]],
                                            mapper=mapper,
                                            optimizer=optimizer,
                                            freeze_core=False,
                                            est_options=est_options,
                                            device='ibmq_montreal'
                                          )

write_list(vqe_results,f'../VQE_results/MG+CO2_ibmq_montreal_4qubit_vqe_results_{int(distances[0]*10)}')
write_list(vqe_problems,f'../VQE_problems/MG+CO2_ibmq_montreal_4qubit_vqe_problems_{int(distances[0]*10)}')