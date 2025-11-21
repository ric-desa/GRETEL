@echo off
:: .\run_all.bat
:: python main.py
:: run_experiments\###.bat

@REM python main.py XPlore_config\NODE\BZR\BZR_NODE-GCN_XPlore++.jsonc
@REM python main.py XPlore_config\NODE\BZR\BZR_NODE-GCN_CF_GNNE.jsonc
@REM python main.py XPlore_config\NODE\BZR\BZR_NODE-GCN_XPlore.jsonc
@REM python main.py XPlore_config\NODE\BZR\BZR_NODE-GCN_XPlore+.jsonc

@REM python main.py XPlore_config\NODE\ENZYMES\ENZYMES_NODE-GCN_XPlore++.jsonc
@REM python main.py XPlore_config\NODE\ENZYMES\ENZYMES_NODE-GCN_CF_GNNE.jsonc
@REM python main.py XPlore_config\NODE\ENZYMES\ENZYMES_NODE-GCN_XPlore+.jsonc
@REM python main.py XPlore_config\NODE\ENZYMES\ENZYMES_NODE-GCN_XPlore.jsonc

python main.py XPlore_config\GRAPH\DBLP\DBLP_GCN_XPlore++.jsonc
python main.py XPlore_config\GRAPH\DBLP\DBLP_GCN_CF-GNNE.jsonc
python main.py XPlore_config\GRAPH\DBLP\DBLP_GCN_CF2.jsonc
python main.py XPlore_config\GRAPH\DBLP\DBLP_GCN_CLEAR.jsonc
python main.py XPlore_config\GRAPH\DBLP\DBLP_GCN_D4Explainer.jsonc
python main.py XPlore_config\GRAPH\DBLP\DBLP_GCN_RSGG-CE.jsonc
python main.py XPlore_config\GRAPH\DBLP\DBLP_GCN_XPlore.jsonc
python main.py XPlore_config\GRAPH\DBLP\DBLP_GCN_XPlore+.jsonc

