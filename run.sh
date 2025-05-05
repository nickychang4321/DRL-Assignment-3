#!/bin/bash
mkdir models
curl -L "https://www.dropbox.com/scl/fi/uc8wtnduftm9bokfx3npk/rainbow_icm_best.pth?rlkey=fyp54296hz38mxbsyp7n7nnb2&st=9x3xtvrc&dl=0" -o "rainbow_icm_best.pth"
mv rainbow_icm_best.pth models/