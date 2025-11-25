@echo off
:: .\run_all.bat
:: python main.py
:: run_experiments\###.bat

@REM python main.py XPlore_config\NODE\ENZYMES\ENZYMES_NODE-GCN_XPlore++.jsonc
@REM python main.py XPlore_config\NODE\ENZYMES\ENZYMES_NODE-GCN_CF_GNNE.jsonc
@REM python main.py XPlore_config\NODE\ENZYMES\ENZYMES_NODE-GCN_XPlore+.jsonc
@REM python main.py XPlore_config\NODE\ENZYMES\ENZYMES_NODE-GCN_XPlore.jsonc

@REM @REM python main.py XPlore_config\NODE\DBLP\DBLP_NODE-GCN_XPlore+.jsonc
@REM @REM python main.py XPlore_config\NODE\DBLP\DBLP_NODE-GCN_CF_GNNE.jsonc
@REM @REM python main.py XPlore_config\NODE\DBLP\DBLP_NODE-GCN_XPlore++.jsonc
@REM @REM python main.py XPlore_config\NODE\DBLP\DBLP_NODE-GCN_XPlore.jsonc

@REM python main.py XPlore_config\NODE\BZR\BZR_NODE-GCN_XPlore.jsonc
@REM python main.py XPlore_config\NODE\BZR\BZR_NODE-GCN_XPlore++.jsonc

@REM python main.py XPlore_config\EMBED\TCR\SF\TCR-5000-28-0.3_GCN_XPlore.jsonc
@REM python main.py XPlore_config\EMBED\TCR\SF\TCR-5000-28-0.3_GCN_CF-GNNE.jsonc
@REM python main.py XPlore_config\EMBED\TCR\SF\TCR-5000-28-0.3_GCN_RSGG.jsonc
@REM python main.py XPlore_config\EMBED\TCR\SF\TCR-5000-28-0.3_GCN_CLEAR.jsonc

python main.py XPlore_config\EMBED\TCR\Wavelet\TCR-5000-28-0.3_GCN_XPlore.jsonc
python main.py XPlore_config\EMBED\TCR\Wavelet\TCR-5000-28-0.3_GCN_CF-GNNE.jsonc
python main.py XPlore_config\EMBED\TCR\Wavelet\TCR-5000-28-0.3_GCN_RSGG.jsonc
python main.py XPlore_config\EMBED\TCR\Wavelet\TCR-5000-28-0.3_GCN_CLEAR.jsonc
