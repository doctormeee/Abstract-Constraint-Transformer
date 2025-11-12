#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#########################################################################
##   Abstract Constraint Transformer (ACT) - Path Configuration        ##
##                                                                     ##
##   doctormeeee (https://github.com/doctormeeee) and contributors     ##
##   Copyright (C) 2024-2025                                           ##
##                                                                     ##
##   This module provides unified path configuration for all ACT       ##
##   modules, ensuring consistent imports regardless of file location. ##
##                                                                     ##
#########################################################################

import os
import sys

def setup_act_paths():
    current_file = os.path.abspath(__file__)
    verifier_root = os.path.dirname(current_file)
    if verifier_root not in sys.path:
        sys.path.insert(0, verifier_root)
    return verifier_root

verifier_root = setup_act_paths()


def setup_gurobi_license():
    """
    Auto-detect and set GRB_LICENSE_FILE environment variable if not already set.
    
    This function searches for gurobi.lic in the project's gurobi/ directory
    and sets the environment variable so Gurobi can find the license file.
    """
    try:
        # If license already configured, do nothing
        if 'GRB_LICENSE_FILE' in os.environ:
            print(f"[ACT] Using existing Gurobi license: {os.environ['GRB_LICENSE_FILE']}")
            return

        if 'ACTHOME' in os.environ:
            license_path = os.path.join(os.environ['ACTHOME'], 'gurobi', 'gurobi.lic')
            print(f"[ACT] Using ACTHOME environment variable: {os.environ['ACTHOME']}")
        else:
            # Use verifier_root to infer project root
            project_root = os.path.dirname(verifier_root)
            license_path = os.path.join(project_root, 'gurobi', 'gurobi.lic')
            print(f"[ACT] Auto-detecting project root from path_config")

        license_path = os.path.abspath(license_path)

        if os.path.exists(license_path):
            os.environ['GRB_LICENSE_FILE'] = license_path
            print(f"[ACT] Gurobi license found and set: {license_path}")
        else:
            print(f"[WARN] Gurobi license not found at: {license_path}")
            print(f"[INFO] Please ensure gurobi.lic is placed in: {os.path.dirname(license_path)}")
    except Exception as e:
        print(f"[WARN] setup_gurobi_license encountered an error: {e}")

setup_gurobi_license()