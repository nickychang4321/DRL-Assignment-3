#!/bin/bash
mkdir models
curl -L "https://www.dropbox.com/scl/fi/50vhbhibsm7r8bc364145/rainbow_icm_best.pth?rlkey=lhvqjg08g8qx5u06setuszl70&st=l53hy3v9&dl=0" -o "rainbow_icm_best.pth"
mv rainbow_icm_best.pth models/