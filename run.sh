#!/bin/bash
mkdir models
curl -L "https://www.dropbox.com/scl/fi/o6anscmghazfjdmeu9b7m/rainbow_icm_episode_1300.pth?rlkey=czto29lgotq01jap5xwnfeepy&st=xxk2cbb2&dl=0" -o "rainbow_icm_episode_1300.pth"
mv rainbow_icm_episode_1300.pth models/