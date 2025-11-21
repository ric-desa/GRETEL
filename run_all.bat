@echo off
:: .\run_all.bat
:: python main.py
:: run_experiments\###.bat

@REM python main.py XPlore_config\NODE\ENZYMES\ENZYMES_NODE-GCN_XPlore++.jsonc
python main.py XPlore_config\NODE\ENZYMES\ENZYMES_NODE-GCN_CF_GNNE.jsonc
python main.py XPlore_config\NODE\ENZYMES\ENZYMES_NODE-GCN_XPlore+.jsonc
python main.py XPlore_config\NODE\ENZYMES\ENZYMES_NODE-GCN_XPlore.jsonc


