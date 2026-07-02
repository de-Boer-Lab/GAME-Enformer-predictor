#!/bin/bash

# This script is designed to run the Enformer predictor in development mode using Apptainer.
# It performs the following steps (script mounting):
# 1. Checks for the existence of the Enformer container image (predictor_enformer.sif).
# 2. Dynamically generates an available IP address and port for the predictor.
# 3. Accepts mandatory matcher IP and port as command-line arguments.
# 4. Mounts the current directory into the container and sets the working directory.
#   It also sets the PYTHONPATH to include the necessary directories for the predictor to function correctly.
# 4. Displays the configuration and starts the predictor using Apptainer.


# 1. Define image name
CONTAINER_IMG="predictor_enformer.sif"

# 2. Check if image exists
if [ ! -f "$CONTAINER_IMG" ]; then
    echo "❌ Error: $CONTAINER_IMG not found."
    echo "    Did you build it? (apptainer build predictor_enformer.sif predictor_enformer.def)"
    exit 1
fi

# 3. Dynamic IP and Port Generation
pred_ip=$(hostname -I | awk '{print $2}')
# Fixed: 'shuf' can sometimes fail if the input range is empty, added fallback or simple logic
pred_port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# 4. MANDATORY Matcher Arguments
matcher_ip=$1
matcher_port=$2

if [[ -z "$matcher_ip" || -z "$matcher_port" ]]; then
    echo "❌ Error: Matcher arguments are mandatory for Enformer."
    echo "Usage: ./dev_run.sh <MATCHER_IP> <MATCHER_PORT>"
    exit 1
fi

echo "=========================================================="
echo "🧪 STARTING ENFORMER DEV MODE"
echo "=========================================================="
echo "   Predictor: http://$pred_ip:$pred_port"
echo "   Matcher:   http://$matcher_ip:$matcher_port"
echo "   Mapping:   $PWD  ---> Container /enformer_GAME"
echo "   Python:    Using /opt/conda/envs/enformer17/bin/python3"
echo "----------------------------------------------------------"

# 5. The Apptainer Command
#    We use the ABSOLUTE path to the python interpreter inside the conda env.
#    This guarantees we use the environment built in the .def file, 
#    not the base system python:3.13.

apptainer exec --nv \
    --bind "$PWD:/enformer_GAME" \
    --pwd /enformer_GAME \
    --env PYTHONPATH="/enformer_GAME:/enformer_GAME/Modules:/enformer_GAME/script_and_utils:$PYTHONPATH" \
    "$CONTAINER_IMG" \
    /opt/conda/envs/enformer17/bin/python3 /enformer_GAME/enformer_predictor_rest_api.py "$pred_ip" "$pred_port" "$matcher_ip" "$matcher_port"