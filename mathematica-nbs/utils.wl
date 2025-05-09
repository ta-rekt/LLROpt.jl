(* ::Package:: *)

BeginPackage["utils`"]

GetData::usage="Loads data from julia optimization output"
DeepMerge::usage="Merges a list of associations"

Begin["Private`"]

DeepMerge[list_]:=Merge[list,(*value-merger:if all sub-values are Associations,merge again,otherwise keep the first (or use something else)*)
	If[AllTrue[#,AssociationQ],DeepMerge[#],First[#]]&
];

GetData[dirname_]:=Module[{projectdir, dir, datafiles, paramfile, rawData, rawParams, params, data},
	projectdir = ".julia/dev/LLROpt";
	dir = FileNameJoin[{$HomeDirectory}, projectdir, "data", dirname];
	datafiles = FileNames["ir_full_mix_vs_pure_pareto_pid*.json", dir];
	paramfile = FileNames["params.json", dir];
	rawData = Import[#,"RawJSON"]&/@datafiles;   (*list of Associations*)
	rawParams = Import[#,"RawJSON"]&/@paramfile;
	data = DeepMerge[rawData];
	params = DeepMerge[rawParams];
	{data, params}
]

End[]    (* end Private context *)
EndPackage[]
