#!/bin/bash

# ICP Only, ICP + RANSAC, ICP + FGR
python3 -m icp_operating_range_surface --local_registration --multi_scale
python3 -m icp_operating_range_surface --global_registration --local_registration --ransac --multi_scale
python3 -m icp_operating_range_surface --global_registration --local_registration --fgr --multi_scale
python3 -m icp_operating_range_surface --total_figures