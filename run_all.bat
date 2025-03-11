@echo off
:: python main.py 

python main.py CF-GNN-EXT_config\GRAPH\AIDS\AIDS_GCN_CLEAR.jsonc
python main.py CF-GNN-EXT_config\GRAPH\AIDS\AIDS_GCN_CF2.jsonc
python main.py CF-GNN-EXT_config\GRAPH\AIDS\AIDS_GCN_RSGG-CE.jsonc

python main.py CF-GNN-EXT_config\GRAPH\BBBP\BBBP_GCN_CLEAR.jsonc
python main.py CF-GNN-EXT_config\GRAPH\BBBP\BBBP_GCN_RSGG-CE.jsonc
python main.py CF-GNN-EXT_config\GRAPH\BBBP\BBBP_GCN_CF2.jsonc
python main.py CF-GNN-EXT_config\GRAPH\BBBP\BBBP_GCN_iRand.jsonc

python main.py CF-GNN-EXT_config\GRAPH\COLORS-3\COLORS-3_GCN-CF-GNNE-EXT.jsonc
python main.py CF-GNN-EXT_config\GRAPH\COLORS-3\COLORS-3_GCN-CF-GNNE-EXT+.jsonc
python main.py CF-GNN-EXT_config\GRAPH\COLORS-3\COLORS-3_GCN-CF-GNNE-EXT++.jsonc
python main.py CF-GNN-EXT_config\GRAPH\COLORS-3\COLORS-3_GCN-CF-GNNE.jsonc
python main.py CF-GNN-EXT_config\GRAPH\COLORS-3\COLORS-3_GCN-CF2.jsonc
python main.py CF-GNN-EXT_config\GRAPH\COLORS-3\COLORS-3_GCN-iRand.jsonc

python main.py CF-GNN-EXT_config\GRAPH\TCR\TCR-5000-28-0.3_GCN_D4Explainer.jsonc
python main.py CF-GNN-EXT_config\GRAPH\TG\TG_GCN_D4Explainer.jsonc
python main.py CF-GNN-EXT_config\GRAPH\BAS\BAS_GCN_D4Explainer.jsonc
python main.py CF-GNN-EXT_config\GRAPH\BZR\BZR_GCN_D4Explainer.jsonc
python main.py CF-GNN-EXT_config\GRAPH\COX2\COX2_GCN_D4Explainer.json
python main.py CF-GNN-EXT_config\GRAPH\AIDS\AIDS_GCN_D4Explainer.jsonc
python main.py CF-GNN-EXT_config\GRAPH\BBBP\BBBP_GCN_D4Explainer.jsonc