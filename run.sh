#!/bin/bash
mkdir models
curl -L "https://www.dropbox.com/scl/fi/56d75krqktfrahdtcm0zh/rainbow_icm_episode_9600.pth?rlkey=bnhv0y6eaq4fbfnkj6ei7ouwq&st=lml9ixv4&dl=0" -o "rainbow_icm_best.pth"
mv rainbow_icm_best.pth models/