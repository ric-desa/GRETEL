:: run_experiments\PROTEINS_full.bat
:: python main.py

@REM python main.py XPlore_config\GRAPH\PROTEINS_full\PROTEINS_full_GCN_CF-GNNE.jsonc
@REM python main.py XPlore_config\GRAPH\PROTEINS_full\PROTEINS_full_GCN_CF2.jsonc
python main.py XPlore_config\GRAPH\PROTEINS_full\PROTEINS_full_GCN_CLEAR.jsonc
python main.py XPlore_config\GRAPH\PROTEINS_full\PROTEINS_full_GCN_D4Explainer.jsonc
python main.py XPlore_config\GRAPH\PROTEINS_full\PROTEINS_full_GCN_iRand.jsonc
@REM python main.py XPlore_config\GRAPH\PROTEINS_full\PROTEINS_full_GCN_RSGG-CE.jsonc
@REM python main.py XPlore_config\GRAPH\PROTEINS_full\PROTEINS_full_GCN_XPlore.jsonc
@REM python main.py XPlore_config\GRAPH\PROTEINS_full\PROTEINS_full_GCN_XPlore+.jsonc
@REM python main.py XPlore_config\GRAPH\PROTEINS_full\PROTEINS_full_GCN_XPlore++.jsonc