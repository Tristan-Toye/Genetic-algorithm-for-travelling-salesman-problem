

#########################################################
tour750.csv
START SECOND CORE
index tsp: 144   best: 165711.13785701763  
#########################################################




#########################################################
tour750.csv
START SECOND CORE
index tsp: 144   best: 164602.60711434155  
#########################################################


#########################################################
tour750.csv
START SECOND CORE
index tsp: 133   best: 166534.5537046534  
#########################################################





750-v2:
best_distance 164182.31444



50-v1 & v3 & v5 & v6:
 best: 25440.68402544451 
50-v7
 best_distance 25434.9184

100-v1
best_distance 78667.24588
100-v2
best_distance 78201.26345

200-v1:
37644.85363
200-v2
best_distance 37512.95288
200-v3
best_distance 36631.91361


500-v1
best_distance 131962.94096
500-v2
best_distance 131883.55039
500-v3
best_distance 130058.92664

1000-v1
best_distance 161770.14482
1000-v2
best_distance 160408.85827

Enter 1 to enable WandB tracking or 0 to disable it: 0
#########################################################
tour750.csv
START SECOND CORE
index tsp: 32    best: 167369.94621177652  
avg offspring:1.7198808416724205         max offspring:10.264589548110962         avg phere: 0.14676185697317123         max phere: 0.21072125434875488          avg elim: 1.7198808416724205    avg get: 0.030376702547073364        max get: 0.06504225730895996    avg opt: 0.032027687839543253
#########################################################
         5460671 function calls (5458248 primitive calls) in 61.751 seconds




#########################################################
tour500.csv
START SECOND CORE
index tsp: 207   best: 131133.21851271694  
avg offspring:1.374061464687477  max offspring:11.3707435131073           avg phere: 0.05070545823101836         max phere: 0.06937551498413086          avg elim: 1.374061464687477     avg get: 0.025442420572474384        max get: 0.05829644203186035    avg opt: 0.025631299639442048
#########################################################
tour750.csv
START SECOND CORE
index tsp: 50    best: 164438.29412431928  
avg offspring:6.083717670440675  max offspring:50.82109832763672          avg phere: 0.14363299846649166         max phere: 0.22227716445922852          avg elim: 6.083717670440675     avg get: 0.034013404846191406        max get: 0.0769040584564209     avg opt: 0.033461173386967136
#########################################################
tour1000.csv
START SECOND CORE
index tsp: 69    best: 162727.00201906622  
avg offspring:3.9426373398822285         max offspring:29.142301321029663         avg phere: 0.3036356525144715          max phere: 0.4406774044036865   avg elim: 3.9426373398822285    avg get: 0.03463818370432094         max get: 0.06824564933776855    avg opt: 0.03949027540285848
#########################################################






change inverse to 10% up
check to remove no dont look bits and double bridge


average run time: 0.14696416488060587

check how long put/get queue takes when element is present


reset: 0.004561424255371094     dis_look: 0.0030775070190429688 phere_look: 0.0025718212127685547       trans_dist: 0.2082228660583496  trans_phere: 0.05523324012756348 multi: 0.004609584808349609      select: 0.004045009613037109
overall time in jit function: 2.7417399883270264

reset: 2.4358599185943604       dis_look: 0.003984212875366211  phere_look: 0.004984617233276367        trans_dist: 0.0060231685638427734       trans_phere: 0.002016305923461914        multi: 0.003991842269897461      select: 0.004999399185180664
overall time in jit function: 3.395855665206909

reset: 0.0030069351196289062    dis_look: 0.008300065994262695  phere_look: 0.0031986236572265625       trans_dist: 0.004074811935424805        trans_phere: 0.00598454475402832 multi: 0.0031003952026367188     select: 0.0029935836791992188
overall time in jit function: 0.059403419494628906




don't update pheremone with repeated answers
send pheremone to second core



maximum_number_of_mutations = 1 => back to previous



# Genetic-Algorithms-Project


local search opt after elim selection:
 1) build list with distance of all edges
 => normalise list (/divide by max)
 2) get variance of this list
 3) if variance is small => delete
 4) if variance is big => 
    5) optimise on the biggest edges
    6) how?


# TODO / TOINVESTIGATE

INDUSTRY PAPERS:
crossover: ER (semi check), OX1 (done, no debug), POS (done, no debug), heuristic (done, no debug)
mutation: SIM, DM, IVM (done, no debug)
local search & others opt:
    - simulated annealing
        - see paper in folder:  Applying_the_genetic_approach_to_simulated_annealing_in_solving_some_NP-hard_problems
    - tabu search (less prio)
    - 2-opt lin
        - 3-opt
        - 4- opt (double bridge)
            - cannot be found (or directly undone) by 3-opt & Lin kernighan (pointed out in original Lin-Kernighan paper)
        - 
        - n-opt
        - if changes are made, put cities with broken bounds in queue and recursively call on them, if no change add nothing to queue (efficient implementation)
    - linn-kernighan
- "production-mode" iterated lin-kernighan
    - removing length restriction on 4-opt moves -> random double bridge move
    - removing randomness in acceptance (only accept when better)

Abuse hyperplane analysis with adjency notation -> we can 'fix' subpaths to constants

seeding:
    - Nearest Neighbour (NN):
            - start from different cities with gredy closed neigbourbour link
            - nearest neighbour with starting from different cities 
        - EV: no fucking clue
        - remove outliers in current approach

IDEAS & PEER REVIEW PAPERS:
    - generating multiple offspring
    - Probability inheriting common subpath dependant on length
    - local search:
        - greedy with restart on not possible (due to infinite)
            - implementation: depth first on ordered distances that terminates when a possible sollution is found
        - A*
    - Ranking selection with exponential decay
        - https://algorithmafternoon.com/genetic/ranked_selection_genetic_algorithm/
    - K-nearest neighbours swap mutation
        - random select -> find k-nearest neighbours -> select from neigbours -> swap pair
    - clustering individuals for population control



- offspring generation expansion:
    - path:
        - PMX
        - order crossover (subtour chunks with only two chunks)
            - OX1: random cut on two points, between stays from one parent, the rest filled up with elements in same sequence of other parent
            - OX2: random select cities from parent2, copy all cities to child from parent1 that are not in selected list, put the random selected elements back but in same order as the mapping occurs in parent 2 (same elements, but different order)
        - POS:
            - select random cities from parent, impose location on child, fill in with sequence of other parent
        -  ER:
            - build edge map and connect edges untill map is empty, then random and continue 
            - upgrade: prio on common edges
            - upgrade: better then random choice when no options
                - https://www.researchgate.net/publication/2527549_The_Traveling_Salesman_and_Sequence_Scheduling_Quality_Solutions_Using_Genetic_Edge_Recombination 
        - cycle crossover operator (difficult to implement, others tried didn't use)
            - inheret position in chain
            - https://onlinelibrary.wiley.com/doi/full/10.1155/2017/7430125
        - alternative position crossover 
            - https://www.researchgate.net/figure/Alternating-position-crossover-AP_fig5_226665831 
    - adjency 
        - alternating edges 
            - https://www.flipstar.com.br/portfolio/paulo/uploads/multiple_crossovers_paper_paulo.pdf
            - need list of available cities to avoid cycles -> other list, ordered to avoid kwadratic time
        - subtour chunks
            - build subtours from alternating parents of length (?) untill complete path, use random to avoid cycles
        - heuristic approach (only for adjency representation):
            - pick a random city as the starting point of the child's tour. Compare the two edges leaving the starting city in the parents and choose the shorter edge.
            Continue to extend the partial tour by choosing the shorter of the two edges in the parents which extend the tour. If the shorter parental edge would
            introudce a cycle into the partial tour, then extend the tour by a random edge. Continue until completion
                - adaptation: choose shorter of parent -> choose first possible element in sorted list of neighbours 
                - adaptation: choose shorter of parent -> larger of parent -> shortes out of 'q' randomly selected cities (that are still possible)
                - same time complexity as random search -> linear (with good datastructures)
        - simulated annealing ?
            - http://wexler.free.fr/library/files/kirkpatrick%20(1983)%20optimization%20by%20simulated%20annealing.pdf
                - original paper where they used it for tsp
            - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=257766 

- mutation:
    - Displacement mutation (DM)
        - take subtour and insert somewhere else in path
    - Exchange mutation (EM)
        - exchange two positions
        - introduction of s-value in individual indicating how many swaps to perform
    - Insertation mutation (ISM)
        - takes a cities, removes, insert somewhere else
    - Simple inversion mutation (SIM)
        - take a subpath and inverse it
        - "The simple inversion mutation operator served as the basis for the 2-opt
        heuristic for the TSP developed by Lin (1965) and is also used in the application of simulated annealing to the TSP (Kirkpatrick et al. 1983)."
    - Inversion mutation (IVM)
        - takes subpath, removes, reverses, inserts somewhere else
    - Scramble mutation (SM)
        - takes a subpath and scrambles randomly the cities

- local search optimiser:
    - 2-opt algorithm of Lin:
        - a->b->c->d --> a->c->b->d if (distance a->c < distance a->b) and (distance b->d < distance c->d)
        - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6767727 
        - 3- opt:
            a->b->c->d->e --> bcd, bdc, cbd, cdb, dbc, dcb
        - n-opt: ...
    - https://www.cs.ubc.ca/~hutter/previous-earg/EmpAlgReadingGroup/TSP-JohMcg97.pdf 
        - comprehensive one
    - https://www.cs.princeton.edu/~bwk/btl.mirror/tsp.pdf 
        - lin + kenighan neigborhoods 
    - https://link.springer.com/chapter/10.1007/bfb0032050
    - https://link.springer.com/chapter/10.1007/BFb0029740 
    - threshold accepting
        - https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.approximation.traveling_salesman.threshold_accepting_tsp.html
        - https://www.sciencedirect.com/science/article/pii/002199919090201B
    - tabu search
        - https://www.iasj.net/iasj/download/c8717a7cde6ba605
        - https://file.scirp.org/Html/3-1040110_19930.htm
    - simulated annealing ?
        - http://wexler.free.fr/library/files/kirkpatrick%20(1983)%20optimization%20by%20simulated%20annealing.pdf
            - original paper where they used it for tsp
        - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=257766 

- comparison paper:
    -INVESTIGATE SIMULATED ANNEALING AND TABU SEARCH MOST  
    - " The best quality solutions were generated using simulated annealing. The performance of tabu search was also satisfactory and importantly – guaranteed stable 'run to run' optimization results. "
    - "The first conclusion is that simulated annealing uniformly outperforms all the other metaheuristics used in the experiment over all test cases. Furthermore, despite the relatively small sample test cases, it also produces better results than the greedy 2-opt heuristic. The
    greedy heuristic is also outperformed by tabu search, and for the smallest test case, it is additionally outperformed by harmony search and quantum annealing. Is has to be recalled that all the algorithms were stopped after the 100 s. of runtime. This is because runtime
    (along with financial cost) constitutes the most important operational constraint when solving practical optimization tasks.
    It should be noted that simulated annealing (along with quantum annealing and tabu
    search) explores in each iteration exactly one point (i.e. one solution) of the search space.
    It is instructive to investigate the number of solutions effectively processed during the
    100 s. of optimization. As can be seen from Table 2 on the next page, simulated annealing
    processed the largest number of solutions, ranging from 12 million for the smallest test cases to 3,5 million for the largest test cases."

- random, maybe interesting:
    - https://leeds-faculty.colorado.edu/glover/Publications/TSP.pdf

- seeding (smart initialisation)
    - population variance :
        - https://link.springer.com/article/10.1007/BF00203625 
    - comparison paper:
        - Nearest Neighbour (NN):
            - start from different cities with gredy closed neigbourbour link
            - nearest neighbour with starting from different cities 
        - EV: no fucking clue
        - remove outliers in current approach
        - CONCLUSION: NN and EV
        - "it can be
        observed that the best quality solutions are generated using NN technique and the
        worst quality solutions are generated by random population initialization. A good
        population seeding should maintain a healthy balance between the best and the worst
        solutions it generate. In view of this, the EV technique attracts the attention because it
        generates the solution with good balance between the best and worst solutions
        generated"

        - "The population seeding
        technique that can generate solutions with high convergence diversity can explore
        much more area of the search space on comparing with the lesser one. Table II
        presents the convergence diversity and nearest neighbor ratio for solutions generated
        using different population seeding techniques. From the table values, it is understood
        the random and sorted population techniques has better population diversity than
        other techniques as in Figure 3. But, the effectiveness of the convergence diversity of
        solutions can be measured along the quality of best and worst solutions. With this
        measure, the EV technique has much good convergence diversity with relatively high
        quality best and worst solutions."


- https://legacy.cs.indiana.edu/~vgucht/Genetic_Algorithms_for_the_Travelling_Salesman+Problem.pdf 
    - concludeds that the distance is a bad indicator to select and talks about "hyperplane analysis"
- https://cig.fi.upm.es/wp-content/uploads/2024/01/Genetic-algorithms-for-the-travelling-salesman-problem-A-review-of-representations-and-operators.pdf 
    - good overview of most crossover operations
    - path notation:
        - PMX < CX
    - "
        In this section we have included different crossover and mutation operators that had been developed for the dominated path representation. The
        majority of the work in which the optimal permutation is obtained uses this
        representation. However, from a historic point of view, the detection of the
        problems done by Grefenstette et al. (1985), problems that appear with this
        representation in hyperplans analysis, are those that have caused the introduction of two new representations (ordinal and adjacency) which offer some
        of improvements over the path representation.
        According to Grefenstette et al. (1985):
        ::: there is a problem in applying the hyperplane analysis of GA’s to this
        representation. The definition of a hyperplane is unclear in this representation. For example (a,,,,) appears to be a first order hyperplane, but
        aire
        152 P. LARRANAGA ET AL. ˜
        it contains the entire space. The problem is that in this representation,
        the semantics of an allele in a given position depends on the surrounding alleles. Intuitively, we hope that GA’s will tend to construct good
        solutions by identifying good building blocks and eventually combining
        these to get larger building blocks. For the TSP, the basic building blocks
        are edges. Larger building blocks correspond to larger subtours. The path
        representation does not lend itself to the description of edges and longer
        subtours in ways which are useful to the GA.
    "
    - Best ones for path: ER, OX1, POS, OX2, mutation: DM, IVM and ISM
    - speed: ER, PMX, OX1, POS, mutation : SIM and SM
    - overal: ER, OX1, POS
    - advantage of adjency:
        - hyperplane analysis
            - A hyperplane (actually subcube) is a subset of Ω = {0,1}^(n), where the values of some bits are fixed and other are free to vary
    - adjency: overall worse performance

        

- simulated annealing:
    - In each step of this algorithm, an atom is given a small random
    displacement and the resulting change,
    AE, in the energy of the system is computed. If AE s 0, the displacement is
    accepted, and the configuration with the
    displaced atom is used as the starting
    point of the next step. The case AE > 0
    is treated probabilistically: the probability that the configuration is accepted is
    P(AE) = exp(-AE/kB1).
    - Random numbers uniformly distributed in the interval
    (0,1) are a convenient means of implementing the random part of the algorithm.
    One such number is selected and compared with P(AE). If it is less than
    P(AE), the new configuration is retained; if not, the original configuration is used to start the next step. 

# Possible improvements
- random greedy heuristics
- clustering local optimiser
- we find subclusters of sizer greate then 'parameter'
    - then we do local search optimisation between end points 