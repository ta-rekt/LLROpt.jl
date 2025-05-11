(* ::Package:: *)

BeginPackage["ctmc`"]

RateMatrix::usage = "Builds a rate matrix from edges and weights"

SteadyStateProbabilities::usage = "Steady state probabilities of a CTMC"

VarianceRate::usage = "Rate of change of variance of transcription activity"

EPss::usage = "Entropy production rate at steady state"

MeanFirstPassageTimes::usage = "Mean first passage times"

VarianceFirstPassageTimes::usage = "Variance of first passage times"

LaplaceTransformFPT::usage = "Laplace transform of first passage time density between any two states"

LaplaceTransformToSetByState::usage = "Laplace transform of FPT density from states to a hitting set"

LogSpecificity::usage = "Specificity of a circuit"

Begin["`Private`"]

(* Method of moments estimator *)

(*Function to generate rate matrix from an edge list and weights*)

RateMatrix[edges_List, weights_List, n_Integer] :=
    Module[{Q},
        Q = ConstantArray[0, {n, n}];
        Do[Q[[edges[[i, 1]], edges[[i, 2]]]] = weights[[i]], {i, Length[
            edges]}];
        Do[Q[[i, i]] = -Total[Q[[i, All]]], {i, n}];
        Q
    ];

(*Function to compute steady-state probabilities*)

SteadyStateProbabilities[Q_] :=
    Module[{n, A, b, \[Pi]},
        n = Length[Q];
        A = Append[Transpose[Q], ConstantArray[1, n]];
        b = Append[ConstantArray[0, n], 1];
        \[Pi] = LinearSolve[A, b];
        \[Pi]
    ];

(*Rate of change of variance of transcription activity*)

VarianceRate[Q_, a_] :=
    Module[{Z, pi, nstates, Pss},
        nstates = Length[Q];
        pi = SteadyStateProbabilities[Q];
        (*Check if a is a valid indicator function*)
        If[!VectorQ[a] || Length[a] != nstates,
            Return["Error: a must be a valid indicator function"]
        ];
        (*Check if the length of a matches the number of states in Q*)
            
        If[Length[a] != nstates,
            Return["Error: Length of a must match the number of states in Q"
                ]
        ];
        (*Check if the stationary distribution is valid*)
        If[!VectorQ[pi] || Length[pi] != nstates || !AllTrue[pi, # >=
             0&] || Total[pi] != 1,
            Return["Error: pi must be a valid stationary distribution"
                ]
        ];
        Pss = Table[pi, {i, 1, nstates}];
        Z = Inverse[Pss - Q] - Pss;
        2 * Sum[a[[i]] * pi[[i]] * Sum[a[[j]] * Z[[i, j]], {j, 1, nstates
            }], {i, 1, nstates}]
    ];

(*EPssSymbolic[Q] returns the symbolic entropy\[Hyphen]production rate*)

EPss[Q_?MatrixQ] :=
    Module[
        {n = Length[Q], ss, sol, expr}
        ,(*reuse the \[Pi]\[Hyphen]symbols*)
        ss = Array[Symbol["ss" <> ToString[#]]&, n];
        sol = Solve[Join[Table[Sum[ss[[k]] Q[[k, j]], {k, 1, n}] == 0,
             {j, 1, n}], {Total[ss] == 1}], ss];
(*build the double sum \[Sum]_{i,j}[ss_i Q_{ij}-ss_j Q_{ji}] Log[(ss_i Q_{ij})/(ss_j Q_{ji})]
    
    *)
        expr = Sum[(ss[[i]] * Q[[i, j]] - ss[[j]] * Q[[j, i]]) * Log[
            (ss[[i]] * Q[[i, j]]) / (ss[[j]] * Q[[j, i]])], {i, 1, n}, {j, 1, n}]
            ;
        (*substitute in \[Pi]-solution,divide by 2,and simplify*)
        Simplify[(expr /. sol[[1]]) / 2]
    ];

(*(*Function to compute mean first passage times*)
MeanFirstPassageTimes[Q_]:=Module[{n,Z,T},n=Length[Q];
ss = SteadyStateProbabilities[Q];
Pss = Table[ss, {i,n}];
Z=Inverse[Pss-Q]-Pss;
T=Table[(Z[[j,j]]-Z[[i,j]])/ss[[j]],{i,n},{j,n}];
Simplify[T]];*)

MeanFirstPassageTimes[Q_] :=
    Module[{n, T, subIdxs, Qsub, rhs, x},
        n = Length[Q];
        T = ConstantArray[0, {n, n}];(*to hold T_{i->j} for i,j=1..n*)
            
        Do[
            If[i == j,(*If same state,MFPT=0 by definition*)
                T[[i, j]] = 0
                ,(*Else i!=j,solve the submatrix system:*)
                subIdxs = Delete[Range[n], j];(*The set of states except j
                    *)
                Qsub = Q[[subIdxs, subIdxs]];(*Q^(j):remove row j,col j
                    *)
                rhs = -ConstantArray[1, n - 1];(*-1 vector*)(*Solve Qsub x=-1
                    *)
                x = LinearSolve[Qsub, rhs];
                (*x maps subIdxs->T_{(subIdxs)->j}.We want the entry for i.
                    *)
                T[[i, j]] = x[[First @ Position[subIdxs, i]]];
            ]
            ,
            {i, 1, n}
            ,
            {j, 1, n}
        ];
        Return[T];
    ];

VarianceFirstPassageTimes[Q_] :=
    Module[{n, T, subIdxs, Qsub, rhs, x},
        n = Length[Q];
        m = MeanFirstPassageTimes[Q];
        T = ConstantArray[0, {n, n}];(*to hold T_{i->j} for i,j=1..n*)
            
        Do[
            If[i == j,(*If same state,MFPT=0 by definition*)
                T[[i, j]] = 0
                ,(*Else i!=j,solve the submatrix system:*)
                subIdxs = Delete[Range[n], j];(*The set of states except j
                    *)
                Qsub = Q[[subIdxs, subIdxs]];(*Q^(j):remove row j,col j
                    *)
                rhs = -2 * m[[subIdxs, j]];(*-1 vector*)(*Solve Qsub x=-2m
                    *)
                x = LinearSolve[Qsub, rhs];
                (*x maps subIdxs->T_{(subIdxs)->j}.We want the entry for i.
                    *)
                T[[i, j]] = x[[First @ Position[subIdxs, i]]] - m[[i,
                     j]] ^ 2;
            ]
            ,
            {i, 1, n}
            ,
            {j, 1, n}
        ];
        Return[T];
    ];

LaplaceTransformFPT[Q_, s_] :=
    Module[{n, T, subIdxs, Qsub, rhs, x},
        n = Length[Q];
        T = ConstantArray[0, {n, n}];
        (*T[[i,j]] will store phi_{i->j}(s)=E[e^{-s T_{i->j}}].*)
        Do[
            If[i == j,(*By definition,time from j to j is 0=>Laplace transform=E[e^{-s*0}]=1
                *)
                T[[i, j]] = 1
                ,(*i!=j:solve the submatrix system (Q+sI)_{S' x S'}*phi_{S'}=-Q_{S',j},using the boundary condition phi_j(s)=1.
                    *)
                subIdxs = Delete[Range[n], j];(*states except j*)
                Qsub = Q[[subIdxs, subIdxs]] - s * IdentityMatrix[n -
                     1];
                (*Build the inhomogeneous RHS=-Q[i,j],for each i in subIdxs
                    *)
                rhs = Table[-Q[[subIdxs[[row]], j]], {row, 1, n - 1}]
                    ;
                x = LinearSolve[Qsub, rhs];
                (*x gives phi_{k->j}(s) for k in subIdxs.We want the component corresponding to i.
                    *)
                T[[i, j]] = x[[First @ Position[subIdxs, i]]];
            ]
            ,
            {i, 1, n}
            ,
            {j, 1, n}
        ];
        Return[FullSimplify[T]];
    ];

LaplaceTransformToSetByState[Q_, start_, hit_List, s_] :=
    Module[
        {n = Length[Q], subStates, m, results, eqMatrix, eqRHS, hj, sol,
             iPos}
        ,
(*If'start' is already in'hit',then the time is 0 with prob=1=>transform=1,
    
    
but specifically we don't "hit h_j first" if j!=that state.We'll handle it below.
    
    *)
        If[MemberQ[hit, start],(*Then we trivially return a vector of dimension Length[hit], where the entry for'start' is 1,and all others 0.
            
            *)
            Return[
                Map[
                    If[# == start,
                        1
                        ,
                        0
                    ]&
                    ,
                    hit
                ]
            ]
        ];
        (*Otherwise,we solve one linear system for each h_j in'hit'.*)
            
        subStates = Complement[Range[n], hit];(*non-absorbing set*)
        m = Length[subStates];
        results =
            Table[
                ((*Build the (Q-sI) submatrix for non-absorbing states.
                    
   *)eqMatrix = Table[Q[[subStates[[i]], subStates[[j]]]], {i, m}, {j,
     m}];
                    Do[eqMatrix[[i, i]] -= s, {i, 1, m}];
                    (*RHS=-Q_{i,h_j},for each i in subStates.*)
                    eqRHS = Table[-Q[[subStates[[i]], hj]], {i, 1, m}
                        ];
                    (*Solve and get phi_i^(j)(s).Then extract the'start' component.
                        *)
                    sol = LinearSolve[eqMatrix, eqRHS];
                    iPos = First @ Position[subStates, start];
                    sol[[iPos]]
                )
                ,
                {hj, hit}(*for each absorbing state in'hit'*) ];
        Return[results];
    ];

(*LogSpecificity[c_,\[Omega]_,\[Gamma]_,rates_,edges_]:= Module[{n,Q, weights,ss,kboff,konb,kuon,kbon,koffu,kuoff,konu},
{kboff,konb,kuon,kbon,koffu,kuoff,konu} = Table[rates[[i]][[1]]/.rates[[i]],{i,1,Length[rates]}];
weights={c*kboff,kuoff,\[Omega]*c*kboff,\[Gamma]*kuoff,c*kbon,kuon,\[Omega]*c*kbon,\[Gamma]*kuon,konb,koffu,konb,konu};
Q = RateMatrix[edges, weights,6];
ss = SteadyStateProbabilities[Q];
Log[ss[[3]]/ss[[5]]] + Log[\[Omega]/\[Gamma]]
];*)

LogSpecificity[c_, \[Omega]_, \[Gamma]_, rates_, edges_] :=
    Module[{n, Q, weights, ss, kboff, konb, kuon, kbon, koffu, kuoff,
         konu, koffb},
        {kboff, konb, kuon, kbon, koffu, kuoff, konu, koffb} = Table[
            rates[[i]][[1]] /. rates[[i]], {i, 1, Length[rates]}];
        weights = {c * kboff, kuoff, \[Omega] * c * kboff, \[Gamma] *
             kuoff, c * kbon, kuon, \[Omega] * c * kbon, \[Gamma] * kuon, konb, koffu,
             konb, konu, koffb, koffb};
        Q = RateMatrix[edges, weights, 6];
        ss = SteadyStateProbabilities[Q];
        Log[ss[[3]] / ss[[5]]] + Log[\[Omega] / \[Gamma]]
    ]

End[] (* end Private context *)

EndPackage[]
