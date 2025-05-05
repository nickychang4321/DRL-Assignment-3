#!/bin/bash
mkdir models
curl -L "https://www.dropbox.com/scl/fi/cus7uxf2s791mamt85gmz/rainbow_icm_best.pth?rlkey=af0zt4j3eay0z4tsstc96fk29&st=0xrhiz5e&dl=0" -o "rainbow_icm_best.pth"
mv rainbow_icm_best.pth models/