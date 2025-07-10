import cProfile

import pstats
import io
import threading
import time

from Debug import PRINT_STAGES

import Reporter
import numpy as np
from Initialisation import Initialisation
from Parameters import Parameters
from SecondCoreProcess import SecondCoreProcess
from TravellingSalesMan import TravellingSalesMan

import wandb
from multiprocessing import  Process, Queue

class r0849537:

	_parameters: Parameters

	def __init__(self, use_wandb, run_name=None):
		self._parameters = Parameters()	
		self.use_wandb = use_wandb
		self.reporter = Reporter.Reporter(self.__class__.__name__)

		if self.use_wandb:
			wandb.init(
				project="Genetic-Algorithms-Project",
				entity="dag-malstaf-ku-leuven",
				name=run_name,
				config={
					"population_size": self._parameters.get_population_size(),
					"offspring_size": self._parameters.get_offspring_size(),
					"offspring_per_recombination": self._parameters.get_offspring_size(),
					"k_selection": self._parameters.get_k_selection(),
					"k_elimination": self._parameters.get_k_elimination(),
					"mutation_rate": self._parameters.get_mutation_rate()
				}
			)

	def optimize(self, filename, index = 0):
		print(filename)
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		self._parameters.set_number_of_cities(distanceMatrix.shape[0])

		#parallel future
		#time how long it takes for 1000 cities
		ordered_arg_distance_matrix = np.argsort(distanceMatrix, axis = 1,).astype(np.int16, casting='same_kind')
		#end
		pheromone_matrix = np.zeros_like(distanceMatrix, dtype= np.float64)
		initialisation = Initialisation(self._parameters, distanceMatrix, ordered_arg_distance_matrix, pheromone_matrix)
		
		population = initialisation.get_initial_population()
		


		queue_second_core_to_tsp = Queue()
		queue_tsp_to_second_core = Queue()
		second_core = SecondCoreProcess(self.reporter,initialisation,population.get_population_exposed()[:3],distanceMatrix, ordered_arg_distance_matrix, queue_second_core_to_tsp = queue_second_core_to_tsp, queue_tsp_to_second_core = queue_tsp_to_second_core)
		#second_core.run()
		second_core_process = Process(target= second_core.run, daemon=True)
		tmp =threading.Thread(target = self._build_second_process, kwargs = {
			 	"second_core_process": second_core_process
		})
		tmp.start()
		tsm = TravellingSalesMan(self._parameters, distanceMatrix, ordered_arg_distance_matrix, initialisation, pheromone_matrix,queue_second_core_to_tsp =  queue_second_core_to_tsp, queue_tsp_to_second_core= queue_tsp_to_second_core )

		yourConvergenceTestsHere = True
		while( yourConvergenceTestsHere ):
			if PRINT_STAGES:
				print("#########################################################")			
				print("run tsp")
			try:
				meanObjective, bestObjective, bestSolution = tsm.run( population)
			except Exception as e:
				print(e)
			if PRINT_STAGES:
				print("end run tsp")
				print("#########################################################")
			if self.use_wandb:
				wandb.log({
					"mean_distance": meanObjective,
					"best_distance": bestObjective
				})
			
			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			
			if PRINT_STAGES:
				print(f"time left: {timeLeft}")
				print("#########################################################")
			if timeLeft < 0:
				

				if tsm._best_objective < 25440:
					print(population.get_population_exposed()[0])

				with open("results.csv", "a") as outFile:
					outFile.write("\n" + str(index) + ","+str(tsm._best_objective)+ "," + str(tsm._mean_objective))
				queue_second_core_to_tsp.close()
				queue_tsp_to_second_core.close()
				second_core_process.terminate()
				time.sleep(0.1)
				second_core_process.close()
				break

		return 0
	
	def _build_Ordered_Distance_matrix( self, distance_matrix: np.ndarray, ordered_distance_matrix: np.ndarray):
		for index in distance_matrix:
			ordered_distance_matrix[index] = np.sort(distance_matrix[index])

	def _build_second_process(self, second_core_process: Process):

		second_core_process.start()


if __name__ == "__main__":


	for i in range(500):
		student = r0849537(False, "")
		student.optimize("tour50.csv", i)

	
	exit(0)