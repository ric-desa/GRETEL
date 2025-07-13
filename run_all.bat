@echo off
:: .\run_all.bat
:: python main.py

:: run_experiments\ENZYMES.bat

@REM python main.py XPlore_config\GRAPH\TRIANGLES\TRIANGLES_GCN_XPlore++.jsonc
@REM python main.py XPlore_config\GRAPH\TRIANGLES\TRIANGLES_GCN_RSGG-CE.jsonc

python main.py XPlore_config\GRAPH\COLORS-3\COLORS-3_GCN-CLEAR.jsonc
python main.py XPlore_config\GRAPH\COLORS-3\COLORS-3_GCN-D4Explainer.jsonc
python main.py XPlore_config\GRAPH\COLORS-3\COLORS-3_GCN-RSGG-CE.jsonc