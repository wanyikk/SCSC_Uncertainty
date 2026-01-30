@echo off
REM 单卡训练（推荐、最兼容 Windows）
python basicsr/train.py ^
    -opt options/train/train_ddcolor.yml ^
    --auto_resume ^
    --launcher none

pause
